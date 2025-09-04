# Python Code Collection

## ai_models.py

```python
import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import onnx
import onnxruntime as ort
from onnxmltools import convert_xgboost, convert_sklearn, convert_lightgbm
from onnxconverter_common import FloatTensorType, convert_float_to_float16
from transformers import pipeline, TimeSeriesTransformerModel, TimeSeriesTransformerConfig, DistilBertTokenizer, DistilBertForSequenceClassification
import logging
from scipy.stats import ks_2samp
import os
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from utils import save_data, check_hardware, make_get_request
from datetime import timedelta, datetime
import asyncio
import aiohttp
from textblob import TextBlob
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import traceback
from data_acquisition import fetch_sentiment_data                                                     

# 全局特徵常量，避免重複定義
FEATURES = ['close', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Stoch_k', 'ADX', 'BB_upper', 'BB_lower', 'EMA_12', 'EMA_26', 'ATR', 'Ichimoku_tenkan', 'Ichimoku_kijun', 'Ichimoku_span_a', 'Ichimoku_span_b', 'Ichimoku_cloud_top', 'fed_funds_rate']

class LSTMModel(nn.Module):
    """LSTM 模型：用於價格預測，捕捉時間序列模式。"""
    def __init__(self, input_size=18, hidden_size=50, num_layers=2, output_size=1, device=torch.device('cpu')):
        """初始化 LSTM 模型，支援多個技術指標和經濟數據的輸入大小。"""
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # 使用 LSTMCell 避免融合操作問題
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(device)

    def forward(self, x):
        # 關鍵邏輯：將輸入數據移動到指定設備，並通過 LSTMCell 和全連接層進行前向傳播。
        # 移除 unsqueeze(1)，因為使用 LSTMCell 無需序列維度（適合單步輸入）
        x = x.to(self.device)  # (batch, input_size)
        # 初始化隱藏狀態和細胞狀態
        h = [torch.zeros(x.size(0), self.hidden_size, device=self.device) for _ in range(self.num_layers)]
        c = [torch.zeros(x.size(0), self.hidden_size, device=self.device) for _ in range(self.num_layers)]
        # 手動堆疊多層 LSTMCell
        for i in range(self.num_layers):
            input_i = x if i == 0 else h[i-1]
            h[i], c[i] = self.lstm_cells[i](input_i, (h[i], c[i]))
        # 使用最後一層的 h 作為輸出
        return self.fc(h[-1])

def validate_data(df: pd.DataFrame, required_features: list) -> bool:
    """驗證輸入數據是否包含必要的特徵欄位。"""
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        logging.error(f"缺少必要特徵: {missing_features}")
        return False
    # 修復: 自動填補 NaN 而非僅記錄錯誤
    df[required_features] = df[required_features].ffill().bfill().fillna(0)
    if df[required_features].isna().any().any():
        logging.error("數據包含 NaN 值，即使填補後")
        return False
    return True

def prepare_data(df: pd.DataFrame, target_col: str = 'close') -> tuple:
    """準備訓練數據：驗證、移除 NaN、拆分 X/y。"""
    if not validate_data(df, FEATURES):
        return None, None, None, None
    X = df[FEATURES].values[:-1]
    y = df[target_col].shift(-1).dropna().values
    if len(X) != len(y) or len(X) == 0:
        logging.error(f"X 和 y 長度不匹配或為空 (target: {target_col})")
        return None, None, None, None
    return train_test_split(X, y, test_size=0.2, random_state=42)

def convert_to_onnx(model, input_shape, model_type: str, model_name: str):
    """通用 ONNX 轉換與量化函數。"""
    os.makedirs('models', exist_ok=True)
    onnx_path = f"models/{model_name}.onnx"
    quantized_path = f"models/{model_name}_quantized.onnx"
    initial_types = [('input', FloatTensorType([None, input_shape]))]
    if model_type == 'xgboost':
        onnx_model = convert_xgboost(model, initial_types=initial_types)
    elif model_type == 'sklearn':
        onnx_model = convert_sklearn(model, initial_types=initial_types)
    elif model_type == 'lightgbm':
        onnx_model = convert_lightgbm(model, initial_types=initial_types)
    else:
        return
    onnx.save(onnx_model, onnx_path)
    model_fp32 = onnx.load(onnx_path)
    model_fp16 = convert_float_to_float16(model_fp32)
    onnx.save(model_fp16, quantized_path)
    logging.info(f"{model_type.upper()} 模型轉換為 ONNX 並量化完成: {model_name}")

def train_lstm_model(df: pd.DataFrame, epochs: int = 50, device=torch.device('cpu')):
    """訓練 LSTM：使用歷史數據訓練，保存為 ONNX 格式。"""
    try:
        X_train, X_test, y_train, y_test = prepare_data(df)
        if X_train is None:
            return None
        model = LSTMModel(input_size=len(FEATURES), device=device)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(torch.tensor(X_train, dtype=torch.float32, device=device))
            loss = criterion(output.squeeze(), torch.tensor(y_train, dtype=torch.float32, device=device))
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), 'models/lstm_model.pth')
        dummy_input = torch.tensor(X_train[:1], dtype=torch.float32)
        torch.onnx.export(model, dummy_input, "models/lstm_model.onnx", input_names=["input"], output_names=["output"], opset_version=11)
        convert_to_onnx(None, len(FEATURES), 'none', 'lstm_model')
        return model
    except Exception as e:
        logging.error(f"LSTM 訓練錯誤: {e}")
        return None

def train_tree_model(df: pd.DataFrame, model_class, model_name: str, target_col: str = 'close', compare_xgb: bool = False):
    """通用樹模型訓練函數：支援 XGBoost/RF/LightGBM。"""
    try:
        X_train, X_test, y_train, y_test = prepare_data(df, target_col)
        if X_train is None:
            return None
        if model_class == xgb.XGBRegressor:
            model = model_class(n_estimators=100, learning_rate=0.1)
            converter = 'xgboost'
        elif model_class == RandomForestRegressor:
            model = model_class(n_estimators=100, random_state=42)
            converter = 'sklearn'
        elif model_class == lgb.LGBMRegressor:
            model = model_class(n_estimators=100, learning_rate=0.1)
            converter = 'lightgbm'
        else:
            return None
        model.fit(X_train, y_train)
        joblib.dump(model, f'models/{model_name}.pkl')
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"{model_name.upper()} 性能：RMSE={rmse:.4f}, R²={r2:.4f}")
        if compare_xgb and model_class != xgb.XGBRegressor:
            xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            xgb_rmse = mean_squared_error(y_test, xgb_pred, squared=False)
            xgb_r2 = r2_score(y_test, xgb_pred)
            logging.info(f"XGBoost 性能：RMSE={xgb_rmse:.4f}, R²={xgb_r2:.4f}")
            logging.info(f"性能比較：{model_name.upper()} RMSE={rmse:.4f} vs XGBoost RMSE={xgb_rmse:.4f}")
        convert_to_onnx(model, len(FEATURES), converter, model_name)
        return model
    except Exception as e:
        logging.error(f"{model_name.upper()} 訓練錯誤: {e}")
        return None

def train_xgboost_model(df: pd.DataFrame):
    return train_tree_model(df, xgb.XGBRegressor, 'xgboost_model')

def train_random_forest_model(df: pd.DataFrame):
    return train_tree_model(df, RandomForestRegressor, 'rf_model')

def train_lightgbm_model(df: pd.DataFrame):
    return train_tree_model(df, lgb.LGBMRegressor, 'lightgbm_model', target_col='ATR', compare_xgb=True)

def train_timeseries_transformer(df: pd.DataFrame, epochs: int = 10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """訓練 TimeSeriesTransformer 模型：用於時間序列預測，保存為 ONNX 格式。"""
    try:
        seq_len = 60
        df_clean = df.dropna(subset=FEATURES + ['close']).copy()
        num_seq = len(df_clean) - seq_len
        if num_seq <= 0:
            logging.warning("數據不足以構造時間序列，無法訓練 TimeSeriesTransformer，返回 None")
            return None
        X = np.array([df_clean[FEATURES].iloc[i:i+seq_len].values for i in range(num_seq)])
        y = df_clean['close'].iloc[seq_len:].values
        if len(X) != len(y):
            logging.error(f"X 和 y 長度不匹配: X={len(X)}, y={len(y)}")
            return None
        if X.shape[1] != seq_len or X.shape[2] != len(FEATURES):
            logging.error(f"輸入數據維度錯誤: 實際形狀={X.shape}, 期望形狀=(*, {seq_len}, {len(FEATURES)})")
            return None
        config = TimeSeriesTransformerConfig(
            input_size=len(FEATURES), time_series_length=seq_len, prediction_length=1, d_model=64
        )
        model = TimeSeriesTransformerModel(config).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        past_time_features = torch.tensor(np.arange(seq_len).repeat(len(X_train)).reshape(len(X_train), seq_len)[:, :, None], dtype=torch.float32, device=device)
        past_observed_mask = torch.ones(len(X_train), seq_len, 1, dtype=torch.float32, device=device)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(past_values=torch.tensor(X_train, dtype=torch.float32, device=device), 
                           past_time_features=past_time_features, 
                           past_observed_mask=past_observed_mask).logits
            loss = criterion(output.squeeze(), torch.tensor(y_train, dtype=torch.float32, device=device))
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), 'models/timeseries_transformer.pth')
        dummy_past_values = torch.tensor(X_train[:1], dtype=torch.float32, device=device)
        dummy_past_time_features = torch.tensor(np.arange(seq_len).repeat(1).reshape(1, seq_len)[:, :, None], dtype=torch.float32, device=device)
        dummy_past_observed_mask = torch.ones(1, seq_len, 1, dtype=torch.float32, device=device)
        torch.onnx.export(model, (dummy_past_values, dummy_past_time_features, dummy_past_observed_mask), "models/timeseries_transformer.onnx", input_names=["past_values", "past_time_features", "past_observed_mask"], output_names=["output"], opset_version=11)
        convert_to_onnx(None, len(FEATURES), 'none', 'timeseries_transformer')
        return model
    except Exception as e:
        logging.error(f"TimeSeriesTransformer 訓練錯誤: {e}, traceback={traceback.format_exc()}")
        return None

def train_distilbert(df: pd.DataFrame, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """訓練 DistilBERT 模型：用於情緒分析，保存為 ONNX 格式。"""
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)
        if 'tweets' not in df.columns or df['tweets'].dropna().empty:
            logging.warning("df缺少'tweets'列或無有效推文數據，使用模擬數據 fallback")
            # 修改：添加 fallback 模擬數據
            texts = ["Positive sample tweet about USDJPY", "Negative sample tweet about Federal Reserve"] * 50
            labels = [1, 0] * 50
        else:
            texts = df['tweets'].dropna().tolist()[:100]
            labels = [1 if TextBlob(text).sentiment.polarity > 0 else 0 for text in texts]
        if not texts or len(texts) != len(labels):
            logging.error("無效的文本或標籤數據")
            return None, None
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        optimizer = Adam(model.parameters(), lr=5e-5)
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(**inputs).logits
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), 'models/distilbert_model.pth')
        dummy_input = {
            'input_ids': tokenizer(['dummy text'], return_tensors="pt", padding=True, truncation=True)['input_ids'].to(device),
            'attention_mask': tokenizer(['dummy text'], return_tensors="pt", padding=True, truncation=True)['attention_mask'].to(device)
        }
        torch.onnx.export(model, (dummy_input['input_ids'], dummy_input['attention_mask']),
                          "models/distilbert_model.onnx", input_names=["input_ids", "attention_mask"], output_names=["output"], opset_version=14)  # 修改 opset_version=14
        convert_to_onnx(None, len(FEATURES), 'none', 'distilbert_model')  # 呼叫通用 ONNX 函數
        return model, tokenizer
    except Exception as e:
        logging.error(f"DistilBERT 訓練錯誤: {e}, traceback={traceback.format_exc()}")
        return None, None

async def predict_sentiment(date: str, db_path: str, config: dict) -> float:
    """情緒分析：先從 DB/CSV 讀取推文，如果無數據再呼叫 fetch_sentiment_data 獲取，計算 polarity 並儲存結果及生成 CSV。"""
    try:
        # 先嘗試從 DB 讀取推文
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
        query = text("SELECT * FROM tweets WHERE date = :date")
        async with engine.connect() as conn:
            result = await conn.execute(query, {'date': date})
            df_tweets = pd.DataFrame(result.fetchall(), columns=result.keys())
        if not df_tweets.empty and 'text' in df_tweets.columns:
            logging.info(f"從 DB 載入 tweets 數據: shape={df_tweets.shape}")
        else:
            # DB 無數據，從 CSV 讀取
            csv_path = Path(config['system_config']['root_dir']) / 'data' / 'tweets.csv'
            if csv_path.exists():
                df_tweets = pd.read_csv(csv_path, parse_dates=['date'])
                df_tweets = df_tweets[df_tweets['date'] == pd.to_datetime(date)]
                if not df_tweets.empty:
                    logging.info(f"從 CSV 載入 tweets 數據: shape={df_tweets.shape}")
                    await save_data(df_tweets, timeframe='1 day', db_path=db_path, data_type='tweets')  # 同步到 DB
                else:
                    logging.warning("CSV 無匹配數據，呼叫 API 獲取")
                    df_tweets = await fetch_sentiment_data(date, db_path, config)
            else:
                logging.warning("無 CSV 檔案，呼叫 API 獲取")
                df_tweets = await fetch_sentiment_data(date, db_path, config)
        
        if df_tweets.empty or 'text' not in df_tweets.columns:
            logging.error("無有效推文數據可用，返回預設值 0.0")
            return 0.0
        
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        if not os.path.exists('models/distilbert_model.pth'):
            logging.warning("DistilBERT模型檔案不存在，嘗試重新訓練")
            model, tokenizer = train_distilbert(df_tweets.rename(columns={'text': 'tweets'}), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if model is None:
                logging.error("DistilBERT訓練失敗，使用TextBlob fallback")
                model = None
        else:
            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
            model.load_state_dict(torch.load('models/distilbert_model.pth'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        polarities = []
        for tweet_text in df_tweets['text']:
            inputs = tokenizer(tweet_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs).logits
            score = torch.softmax(outputs, dim=1)[0][1].item() - torch.softmax(outputs, dim=1)[0][0].item()
            tb_polarity = TextBlob(tweet_text).sentiment.polarity
            combined_score = 0.7 * score + 0.3 * tb_polarity
            polarities.append(combined_score)
        if polarities:
            avg_score = sum(polarities) / len(polarities)
            logging.info(f"計算平均 polarity 分數: {avg_score} (DistilBERT 70%, TextBlob 30%)")
        else:
            avg_score = 0.0
            logging.warning("無推文數據，使用預設值 0.0")
        sentiment_df = pd.DataFrame({'date': [pd.to_datetime(date)], 'sentiment': [avg_score]})
        await save_data(sentiment_df, timeframe='1 day', db_path=db_path, data_type='sentiment')
        # 生成 sentiment.csv
        sentiment_csv_path = Path(config['system_config']['root_dir']) / 'data' / 'sentiment.csv'
        sentiment_df.to_csv(sentiment_csv_path, mode='a', header=not sentiment_csv_path.exists(), index=False)
        logging.info(f"情緒數據已存入 CSV: {sentiment_csv_path}")
        return avg_score
    except Exception as e:
        logging.error(f"情緒分析錯誤: {e}, traceback={traceback.format_exc()}")
        return 0.0

def integrate_sentiment(polarity: float) -> float:
    """整合情緒分數：將 polarity 轉換為決策調整值，並檢查極端情緒。"""
    if abs(polarity) > 0.8:
        logging.warning(f"極端情緒分數: {polarity}，建議暫停交易")
        return 0.0
    if polarity > 0.1:
        return 0.1
    elif polarity < -0.1:
        return -0.1
    return 0.0

def detect_drift(old_data: pd.DataFrame, new_data: pd.DataFrame, threshold: float = 0.05) -> bool:
    """檢測數據漂移：使用 KS 檢驗比較分佈。"""
    stat, p_value = ks_2samp(old_data['close'], new_data['close'])
    return p_value < threshold

def predict(model_path: str, input_data: pd.DataFrame, provider='VitisAIExecutionProvider'):
    """使用 ONNX 模型進行推理，支援 NPU，確保輸入數據類型為 float16。"""
    try:
        session = ort.InferenceSession(model_path, providers=[provider, 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        X = input_data[FEATURES].iloc[-1:].values.astype(np.float16)  # 轉換為 float16
        return session.run(None, {'input': X})[0][0]
    except Exception as e:
        logging.error(f"ONNX 推理錯誤: {e}, traceback={traceback.format_exc()}")
        return 0.0

def update_model(df: pd.DataFrame, model_path: str = 'models', session: str = 'normal', device_config: dict = None) -> dict:
    """更新多模型：個別檢查並訓練或載入模型，根據時段加權預測。"""
    os.makedirs('models', exist_ok=True)
    try:
        model_dir = Path(model_path)
        model_dir.mkdir(exist_ok=True)
        models = {}
        required_features = ['close', 'RSI', 'MACD', 'Stoch_k', 'ADX', 'BB_upper', 'BB_lower', 'EMA_12', 'EMA_26', 'fed_funds_rate']
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            logging.error(f"輸入數據缺少必要欄位: {missing_features}")
            return {}
        old_data = df.iloc[:-1000] if len(df) > 1000 else df
        new_data = df.iloc[-1000:]
        data_drift = detect_drift(old_data, new_data)
        device = device_config.get('lstm', torch.device('cuda' if torch.cuda.is_available() else 'cpu')) if device_config else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df_clean = df[required_features].dropna()
        lstm_path = model_dir / 'lstm_model.pth'
        lstm_onnx_path = model_dir / 'lstm_model_quantized.onnx'
        if not lstm_path.exists() or data_drift:
            models['lstm'] = train_lstm_model(df, device=device)
        else:
            models['lstm'] = lstm_onnx_path
            logging.info("載入現有 LSTM ONNX 模型")
        xgboost_path = model_dir / 'xgboost_model.pkl'
        xgboost_onnx_path = model_dir / 'xgboost_model_quantized.onnx'
        if not xgboost_path.exists() or data_drift:
            models['xgboost'] = train_xgboost_model(df)
        else:
            models['xgboost'] = xgboost_onnx_path
            logging.info("載入現有 XGBoost ONNX 模型")
        rf_path = model_dir / 'rf_model.pkl'
        rf_onnx_path = model_dir / 'rf_model_quantized.onnx'
        if not rf_path.exists() or data_drift:
            models['rf_model'] = train_random_forest_model(df)
        else:
            models['rf_model'] = rf_onnx_path
            logging.info("載入現有 RandomForest ONNX 模型")
        lightgbm_path = model_dir / 'lightgbm_model.pkl'
        lightgbm_onnx_path = model_dir / 'lightgbm_model_quantized.onnx'
        if not lightgbm_path.exists() or data_drift:
            models['lightgbm'] = train_lightgbm_model(df)
        else:
            models['lightgbm'] = lightgbm_onnx_path
            logging.info("載入現有 LightGBM ONNX 模型")
        transformer_path = model_dir / 'timeseries_transformer.pth'
        transformer_onnx_path = model_dir / 'timeseries_transformer_quantized.onnx'
        if not transformer_path.exists() or data_drift:
            models['timeseries_transformer'] = train_timeseries_transformer(df, device=device)
        else:
            models['timeseries_transformer'] = transformer_onnx_path
            logging.info("載入現有 TimeSeriesTransformer ONNX 模型")
        distilbert_path = model_dir / 'distilbert_model.pth'
        distilbert_onnx_path = model_dir / 'distilbert_model_quantized.onnx'
        if not distilbert_path.exists() or data_drift:
            models['distilbert'], _ = train_distilbert(df, device=device)
        else:
            models['distilbert'] = distilbert_onnx_path
            logging.info("載入現有 DistilBERT ONNX 模型")
        weights = {
            'lstm': 0.2, 'xgboost': 0.3, 'rf_model': 0.2, 'lightgbm': 0.2, 'timeseries_transformer': 0.1
        } if session == 'high_volatility' else {
            'lstm': 0.3, 'xgboost': 0.2, 'rf_model': 0.2, 'lightgbm': 0.2, 'timeseries_transformer': 0.1
        }
        def predict_price(input_data: pd.DataFrame):
            if input_data.empty or FEATURES[0] not in input_data.columns:
                logging.error("輸入數據為空或缺少必要欄位")
                return 0.0
            predictions = {}
            for name, model in models.items():
                if name == 'distilbert':
                    continue
                if isinstance(model, str):
                    predictions[name] = predict(model, input_data)
                else:
                    X = input_data[FEATURES].iloc[-1:].values
                    if name == 'lstm' or name == 'timeseries_transformer':
                        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
                        model.eval()
                        with torch.no_grad():
                            predictions[name] = model(X_tensor).item()
                    else:
                        predictions[name] = model.predict(X)[0]
            final_pred = sum(weights[name] * pred for name, pred in predictions.items())
            return final_pred
        models['predict'] = predict_price
        logging.info("多模型更新完成")
        return models
    except Exception as e:
        logging.error(f"多模型更新錯誤: {str(e)}, traceback={traceback.format_exc()}")
        return {}
```

## data_acquisition.py

```python
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import aiohttp
import investpy
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from utils import initialize_db, save_data, fetch_api_data, filter_future_dates, make_get_request
from datetime import datetime, timedelta
import asyncio
import os
from pathlib import Path
import logging
import numpy as np
import traceback
import time

# 設置環境變數以禁用 TensorFlow oneDNN 自訂運算
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

_tweets_cache = {}

# 速率限制器類
class RateLimiter:
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.timestamps = []

    async def wait(self):
        current_time = time.time()
        self.timestamps = [t for t in self.timestamps if current_time - t < self.period]
        if len(self.timestamps) >= self.calls:
            sleep_time = self.period - (current_time - self.timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        self.timestamps.append(time.time())

polygon_rate_limiter = RateLimiter(calls=5, period=60.0)

async def fetch_sentiment_data(date: str, db_path: str, config: dict) -> pd.DataFrame:
    """從 X API 獲取推文並儲存到 SQLite 的 tweets 表。（優化：合併重複的錯誤處理）"""
    try:
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=1)
        start_time = start_date.strftime('%Y-%m-%dT00:00:00Z')
        # 修改：將 end_time 設為比當前時間早 10 秒
        current_time = datetime.utcnow()
        end_time = (current_time - timedelta(seconds=10)).strftime('%Y-%m-%dT%H:%M:%SZ')
        logging.info(f"設置 API 請求時間範圍: start_time={start_time}, end_time={end_time}")
        X_BEARER_TOKEN = config['api_key'].get('x_bearer_token', '')
        if not X_BEARER_TOKEN:
            logging.error("X Bearer Token 未配置")
            return pd.DataFrame()
        # 新增：從 DB 取 max tweet_id 作為 since_id
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT MAX(tweet_id) as max_id FROM tweets WHERE date >= :start_date"), {'start_date': start_date.strftime('%Y-%m-%d')})
            row = result.fetchone()
            since_id = row[0] if row[0] else None
        logging.info(f"從 DB 取 since_id: {since_id or '無'}")
        url = "https://api.x.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
        params = {
            'query': "(USDJPY OR USD/JPY OR 'Federal Reserve') AND lang:en",
            'start_time': start_time,
            'end_time': end_time,
            'max_results': 100,
            'expansions': 'author_id',  # 新增：獲取 user 資訊
            'tweet.fields': 'id,text,created_at',
            'user.fields': 'id,username,verified'
        }
        if since_id:
            params['since_id'] = since_id  # 只取新推文
        tweets = []
        users = []  # 新增：存 users
        total_posts = 0
        next_token = None
        while True:  # 新增：分頁 loop
            if next_token:
                params['next_token'] = next_token
            data = await fetch_api_data(url, headers=headers, params=params)
            if 'data' not in data or not data['data']:
                logging.warning(f"X API 數據為空或格式錯誤: {data}")
                break
            total_posts += len(data['data'])
            if total_posts > 100:  # 限每月配額，避免超
                logging.warning(f"已達 100 Posts 上限，停止分頁")
                break
            for tweet in data['data']:
                tweets.append({
                    'date': end_date,
                    'tweet_id': tweet['id'],
                    'text': tweet['text'],
                    'user_id': tweet['author_id']  # 新增
                })
            # 新增：處理 includes.users
            if 'includes' in data and 'users' in data['includes']:
                for user in data['includes']['users']:
                    users.append({
                        'user_id': user['id'],
                        'username': user['username'],
                        'verified': user.get('verified', False)
                    })
            next_token = data.get('meta', {}).get('next_token')
            if not next_token:
                break
            await asyncio.sleep(1)  # 避免速率過快
        tweets_df = pd.DataFrame(tweets)
        users_df = pd.DataFrame(users).drop_duplicates(subset=['user_id'])  # 去重
        if not tweets_df.empty:
            await save_data(tweets_df, timeframe='1 day', db_path=db_path, data_type='tweets')
            await save_data(users_df, timeframe='1 day', db_path=db_path, data_type='users')  # 新增存 users
            logging.info(f"儲存 {len(tweets_df)} 條推文和 {len(users_df)} 位用戶到 SQLite")
            # 生成 tweets.csv
            tweets_csv_path = Path(config['system_config']['root_dir']) / 'data' / 'tweets.csv'
            tweets_df.to_csv(tweets_csv_path, mode='a', header=not tweets_csv_path.exists(), index=False)
            logging.info(f"推文數據已存入 CSV: {tweets_csv_path}")
        else:
            logging.warning("無推文數據")
        return tweets_df
    except Exception as e:
        logging.error(f"推文數據獲取錯誤: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

def normalize_timeframe(tf: str) -> str:
    mapping = {'1 hour': '1h', '4 hours': '4h', '1 day': '1d'}
    return mapping.get(tf, tf)

async def fetch_data(primary_api: str = 'polygon', backup_apis: list = ['yfinance', 'fcs', 'fixer'], date_range: dict = None, timeframe: str = '1d', db_path: str = "C:\\Trading\\data\\trading_data.db", config: dict = None) -> pd.DataFrame:
    """獲取資料：支援多 API，優先使用 primary，失敗則備用。（優化：合併分批邏輯的條件，減少重複的 logging）"""
    timeframe = normalize_timeframe(timeframe)
    start_date = pd.to_datetime(date_range['start'] if date_range else "2025-01-01")
    end_date = pd.to_datetime(date_range['end'] if date_range else "2025-08-25")
    CSV_PATH = Path(config['system_config']['root_dir']) / 'data' / f'usd_jpy_{timeframe}.csv'
    interval_map = {
        '1min': {'multiplier': 1, 'timespan': 'minute', 'yfinance': '1m', 'fcs': '1min', 'fixer': None},
        '5min': {'multiplier': 5, 'timespan': 'minute', 'yfinance': '5m', 'fcs': '5min', 'fixer': None},
        '1h': {'multiplier': 1, 'timespan': 'hour', 'yfinance': '1h', 'fcs': '1hour', 'fixer': None},
        '4h': {'multiplier': 4, 'timespan': 'hour', 'yfinance': '4h', 'fcs': '4hour', 'fixer': None},
        '1d': {'multiplier': 1, 'timespan': 'day', 'yfinance': '1d', 'fcs': '1day', 'fixer': '1d'}
    }
    if timeframe not in interval_map:
        logging.error(f"無效的時間框架: {timeframe}")
        return pd.DataFrame()
    interval = interval_map[timeframe]
    await initialize_db(db_path)
    try:
        # 使用非同步 SQLAlchemy 連線
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
        query = text("SELECT * FROM ohlc WHERE timeframe = :timeframe AND date BETWEEN :start_date AND :end_date")
        async with engine.connect() as conn:
            result = await conn.execute(query, {'timeframe': timeframe, 'start_date': start_date.strftime('%Y-%m-%d'), 'end_date': end_date.strftime('%Y-%m-%d')})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            df['date'] = pd.to_datetime(df['date'])
            if not df.empty:
                df = filter_future_dates(df)
                df = fill_missing_values(df)
                if 'n' not in df.columns:
                    df['n'] = 0
                if 'vw' not in df.columns:
                    df['vw'] = df['close']
                logging.info(f"Loaded data from SQLite: shape={df.shape}, timeframe={timeframe}")
                return df[['date', 'open', 'high', 'low', 'close', 'volume', 'n', 'vw']]
    except Exception as e:
        logging.error(f"SQLite cache query failed: {str(e)}, timeframe={timeframe}, traceback={traceback.format_exc()}")
    apis = [primary_api] + backup_apis
    df_list = []
    current_start = start_date
    while current_start < end_date:
        if interval['timespan'] == 'minute':
            batch_end = current_start + timedelta(days=7)
        elif interval['timespan'] == 'hour':
            batch_end = current_start + timedelta(days=30)
        else:
            batch_end = current_start + timedelta(days=365)
        batch_end = min(batch_end, end_date)
        batch_range = {'start': current_start.strftime('%Y-%m-%d'), 'end': batch_end.strftime('%Y-%m-%d')}
        for api in apis:
            if api == 'fixer' and interval['fixer'] is None:
                logging.warning(f"Fixer API 不支持 {timeframe} 時間框架，跳過")
                continue
            if api == 'yfinance':
                delta = batch_end - current_start
                if (timeframe == '1min' and delta > timedelta(days=7)) or (timeframe == '5min' and delta > timedelta(days=60)):
                    logging.warning(f"yfinance 不支持 {timeframe} 超過限制，跳過此批次")
                    break
                logging.info(f"Batch fetch from {api}: {batch_range['start']} to {batch_range['end']}, timeframe={timeframe}")
                try:
                    for attempt in range(5):
                        try:
                            ticker = yf.Ticker('USDJPY=X')
                            df = ticker.history(start=batch_range['start'], end=batch_range['end'], interval=interval['yfinance'])
                            if df.empty:
                                logging.warning(f"Yahoo Finance batch empty: timeframe={timeframe}")
                                break
                            df = df.reset_index().rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                            df['date'] = pd.to_datetime(df['date'])
                            df['n'] = 0
                            df['vw'] = df['close']
                            df = fill_missing_values(df)
                            df_list.append(df[['date', 'open', 'high', 'low', 'close', 'volume', 'n', 'vw']])
                            logging.info(f"yfinance data fetched: shape={df.shape}, timeframe={timeframe}")
                            break
                        except Exception as e:
                            if attempt == 4:
                                logging.error(f"Yahoo Finance batch failed: {str(e)}, timeframe={timeframe}, traceback={traceback.format_exc()}")
                                break
                            await asyncio.sleep(2 ** attempt * 4)
                except Exception as e:
                    logging.error(f"{api} API batch failed: {str(e)}, timeframe={timeframe}, traceback={traceback.format_exc()}")
                    continue
            elif api == 'polygon':
                if interval['timespan'] == 'minute':
                    batch_end = current_start + timedelta(days=30)
                elif interval['timespan'] == 'hour':
                    batch_end = current_start + timedelta(days=730)
                else:
                    batch_end = current_start + timedelta(days=730)
                batch_end = min(batch_end, end_date)
                batch_range = {'start': current_start.strftime('%Y-%m-%d'), 'end': batch_end.strftime('%Y-%m-%d')}
                api_key = config['api_key'].get('polygon_api_key', '')
                if not api_key:
                    logging.error("Polygon API key not configured")
                    continue
                limit = 50000
                url = f"https://api.polygon.io/v2/aggs/ticker/C:USDJPY/range/{interval['multiplier']}/{interval['timespan']}/{batch_range['start']}/{batch_range['end']}?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}"
                logging.info(f"Attempting Polygon API request: {url}, timeframe={timeframe}")
                page_count = 0
                batch_results = []
                while url:
                    page_count += 1
                    for attempt in range(3):
                        await polygon_rate_limiter.wait()
                        async with aiohttp.ClientSession() as session:
                            try:
                                async with (await make_get_request(session, url, timeout=10)) as response:
                                    data = await response.json()
                                    if data.get('status') not in ['OK', 'DELAYED']:
                                        logging.warning(f"Polygon API failed: status={data.get('status')}, message={data.get('error', 'Unknown error')}, timeframe={timeframe}")
                                        break
                                    if 'results' not in data or not data['results']:
                                        logging.warning(f"Polygon API batch empty: timeframe={timeframe}")
                                        break
                                    logging.info(f"Polygon API page {page_count}: queryCount={data.get('queryCount', 0)}, resultsCount={data.get('resultsCount', 0)}, request_id={data.get('request_id', 'N/A')}, timeframe={timeframe}")
                                    batch_results.extend(data['results'])
                                    next_url = data.get('next_url')
                                    if next_url:
                                        url = f"{next_url}&apiKey={api_key}"
                                        logging.info(f"Fetching next page: {url}")
                                    else:
                                        url = None
                                    break
                            except Exception as e:
                                logging.warning(f"Polygon request failed on attempt {attempt+1}: {str(e)}")
                                if attempt == 2:
                                    url = None
                                await asyncio.sleep(2 ** attempt * 2)
                if batch_results:
                    df_batch = pd.DataFrame(batch_results)[['t', 'o', 'h', 'l', 'c', 'v', 'n', 'vw']].rename(columns={'t': 'date', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'n': 'n', 'vw': 'vw'})
                    df_batch['date'] = pd.to_datetime(df_batch['date'], unit='ms')
                    df_batch = fill_missing_values(df_batch)
                    df_list.append(df_batch)
                    logging.info(f"Polygon data fetched (all pages): shape={df_batch.shape}, pages={page_count}, timeframe={timeframe}")
                else:
                    logging.warning(f"No results from Polygon for batch: {batch_range['start']} to {batch_range['end']}")
            elif api == 'fcs':
                api_key = config['api_key'].get('fcs_api_key', '')
                if not api_key:
                    logging.error("FCS API key not configured")
                    continue
                url = f"https://fcsapi.com/api-v3/forex/history?symbol=USD/JPY&access_key={api_key}&period={interval['fcs']}&from={batch_range['start']}&to={batch_range['end']}"
                async with aiohttp.ClientSession() as session:
                    async with (await make_get_request(session, url, timeout=10)) as response:
                        data = await response.json()
                        if 'response' not in data or not data['response'] or data.get('code') != 200:
                            logging.warning(f"FCS API batch empty or failed: timeframe={timeframe}")
                            continue
                        df = pd.DataFrame(data['response'])[['datetime', 'open', 'high', 'low', 'close', 'volume']].rename(columns={'datetime': 'date'})
                        df['date'] = pd.to_datetime(df['date'])
                        df['n'] = 0
                        df['vw'] = df['close']
                        df = fill_missing_values(df)
                        df_list.append(df)
                        logging.info(f"FCS data fetched: shape={df.shape}, timeframe={timeframe}")
            elif api == 'fixer':
                api_key = config['api_key'].get('fixer_API_Key', '')
                if not api_key:
                    logging.error("Fixer API key not configured")
                    continue
                url = f"http://data.fixer.io/api/timeseries?access_key={api_key}&start_date={batch_range['start']}&end_date={batch_range['end']}&symbols=USD,JPY"
                async with aiohttp.ClientSession() as session:
                    async with (await make_get_request(session, url, timeout=10)) as response:
                        data = await response.json()
                        if not data.get('success') or 'rates' not in data:
                            logging.warning(f"Fixer API batch empty or failed: timeframe={timeframe}")
                            continue
                        rates = data['rates']
                        df_data = []
                        for date, rate in rates.items():
                            if 'USD' in rate and 'JPY' in rate:
                                usd_jpy = rate['JPY'] / rate['USD']
                                df_data.append({'date': date, 'close': usd_jpy, 'open': usd_jpy, 'high': usd_jpy, 'low': usd_jpy, 'volume': 0, 'n': 0, 'vw': usd_jpy})
                        df = pd.DataFrame(df_data)
                        df['date'] = pd.to_datetime(df['date'])
                        df = fill_missing_values(df)
                        df_list.append(df[['date', 'open', 'high', 'low', 'close', 'volume', 'n', 'vw']])
                        logging.info(f"Fixer data fetched: shape={df.shape}, timeframe={timeframe}")
        current_start = batch_end
    if df_list:
        df = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['date'])
        df = filter_future_dates(df)
        df = fill_missing_values(df)
        await save_data(df, timeframe, db_path, data_type='ohlc')
        if not os.path.exists(CSV_PATH.parent):
            os.makedirs(CSV_PATH.parent)
        df.to_csv(CSV_PATH, index=False)
        logging.info(f"Successfully fetched data: shape={df.shape}, timeframe={timeframe}")
        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'n', 'vw']]
    logging.error(f"All APIs and CSV fallback failed, timeframe={timeframe}")
    return pd.DataFrame()

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """填補缺失值：使用前向填補法，確保數據連續性。（優化：合併數值欄位列表，只做 ffill/bfill，依賴 DB default 填 0）"""
    if df.empty:
        return df
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'n', 'vw', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'ATR', 'Stoch_k', 'ADX', 'Ichimoku_tenkan', 'Ichimoku_kijun', 'Ichimoku_span_a', 'Ichimoku_span_b', 'Ichimoku_cloud_top', 'BB_upper', 'BB_middle', 'BB_lower', 'EMA_12', 'EMA_26']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    if 'fed_funds_rate' in df.columns:
        df['fed_funds_rate'] = df['fed_funds_rate'].ffill().bfill()
    logging.info("Missing values filled")
    return df

async def compute_indicators(df: pd.DataFrame, db_path: str, timeframe: str, config: dict = None) -> pd.DataFrame:
    """計算技術指標：使用多線程計算 RSI, MACD 等，支援分批並直接更新 DB。（優化：移除 missing_indicators 檢查，總是計算所有；合併 calc_functions 為字典）"""
    try:
        min_data_length = {'1min': 200, '5min': 150, '1h': 100, '4h': 60, '1d': 60}
        required_length = min_data_length.get(timeframe, 100)
        if len(df) < required_length:
            logging.warning(f"數據長度不足 ({len(df)} < {required_length}) for timeframe {timeframe}，跳過指標計算")
            return df
        df['timeframe'] = timeframe
        # 使用非同步 SQLAlchemy 連線
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
        query = text("SELECT * FROM ohlc WHERE timeframe = :timeframe AND date >= :min_date")
        async with engine.connect() as conn:
            result = await conn.execute(query, {'timeframe': timeframe, 'min_date': df['date'].min().strftime('%Y-%m-%d')})
            db_df = pd.DataFrame(result.fetchall(), columns=result.keys())
            db_df['date'] = pd.to_datetime(db_df['date'])
        if not db_df.empty:
            df = df.merge(db_df, on=['date', 'timeframe'], how='left', suffixes=('', '_db'))
            for col in df.columns:
                if col.endswith('_db') and col.replace('_db', '') in df.columns:
                    df[col.replace('_db', '')] = df[col.replace('_db', '')].combine_first(df[col])
            df = df.drop(columns=[col for col in df.columns if col.endswith('_db')])
            logging.info(f"合併 DB 資料完成")
        batch_size = 10000 if timeframe in ['1min', '5min'] else None
        df_list = [df[i:i+batch_size] for i in range(0, len(df), batch_size)] if batch_size and len(df) > batch_size else [df]
        result_dfs = []
        for batch_df in df_list:
            calc_functions = {
                'RSI': ta.rsi(batch_df['close'], length=14),
                'MACD': ta.macd(batch_df['close']),
                'ATR': ta.atr(batch_df['high'], batch_df['low'], batch_df['close'], length=14),
                'Stoch_k': ta.stoch(batch_df['high'], batch_df['low'], batch_df['close'], k=14, d=3),
                'ADX': ta.adx(batch_df['high'], batch_df['low'], batch_df['close'], length=14),
                'Ichimoku': ta.ichimoku(batch_df['high'], batch_df['low'], batch_df['close'], tenkan=9, kijun=26, senkou=52)[0],
                'BB': ta.bbands(batch_df['close'], length=20, std=2),
                'EMA_12': ta.ema(batch_df['close'], length=12),
                'EMA_26': ta.ema(batch_df['close'], length=26)
            }
            batch_df = batch_df.copy()
            batch_df['RSI'] = calc_functions.get('RSI', pd.Series(np.nan, index=batch_df.index))
            macd = calc_functions.get('MACD')
            if macd is not None:
                batch_df[['MACD', 'MACD_signal', 'MACD_hist']] = macd[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']]
            batch_df['ATR'] = calc_functions.get('ATR', pd.Series(np.nan, index=batch_df.index))
            stoch = calc_functions.get('Stoch_k')
            if stoch is not None:
                batch_df['Stoch_k'] = stoch['STOCHk_14_3_3']
            adx = calc_functions.get('ADX')
            if adx is not None:
                batch_df['ADX'] = adx['ADX_14']
            ich = calc_functions.get('Ichimoku')
            if ich is not None:
                batch_df[['Ichimoku_tenkan', 'Ichimoku_kijun', 'Ichimoku_span_a', 'Ichimoku_span_b']] = ich[['ITS_9', 'IKS_26', 'ISA_9', 'ISB_26']]
                batch_df['Ichimoku_cloud_top'] = batch_df[['Ichimoku_span_a', 'Ichimoku_span_b']].max(axis=1)
            bb = calc_functions.get('BB')
            if bb is not None:
                batch_df[['BB_upper', 'BB_middle', 'BB_lower']] = bb[['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']]
            batch_df['EMA_12'] = calc_functions.get('EMA_12', pd.Series(np.nan, index=batch_df.index))
            batch_df['EMA_26'] = calc_functions.get('EMA_26', pd.Series(np.nan, index=batch_df.index))
            batch_df = fill_missing_values(batch_df)
            result_dfs.append(batch_df)
        df = pd.concat(result_dfs, ignore_index=True).drop_duplicates(subset=['date'])
        await save_data(df, timeframe, db_path, data_type='ohlc')
        logging.info(f"指標數據已存入 DB: timeframe={timeframe}")
        indicators_csv_path = Path(config['system_config']['root_dir']) / 'data' / f'usd_jpy_{timeframe}_indicators.csv'
        df.to_csv(indicators_csv_path, index=False)
        logging.info(f"指標數據已存入 CSV: {indicators_csv_path}")
        return df
    except Exception as e:
        logging.error(f"指標計算錯誤: {str(e)}, timeframe={timeframe}, traceback={traceback.format_exc()}")
        return df

async def import_from_csv(file_name: str, db_path: str, timeframe: str):
    """從 CSV 檔案匯入資料到 DB 的 ohlc 表格，忽略 _db 欄位。"""
    try:
        df = pd.read_csv(file_name, parse_dates=['date'])
        # 忽略 _db 欄位
        df = df[[col for col in df.columns if not col.endswith('_db')]]
        df['timeframe'] = timeframe
        # 填補缺失
        df = fill_missing_values(df)
        # 儲存到 DB
        await save_data(df, timeframe, db_path, data_type='ohlc')
        logging.info(f"從 CSV {file_name} 匯入 {len(df)} 行資料到 DB: timeframe={timeframe}")
        return True
    except Exception as e:
        logging.error(f"從 CSV 匯入失敗: {str(e)}, traceback={traceback.format_exc()}")
        return False

async def fetch_economic_calendar(date_range: dict, db_path: str, config: dict) -> pd.DataFrame:
    """獲取經濟日曆數據並儲存到 SQLite，新增 FRED API 獲取聯邦基金利率。（優化：合併 API 錯誤處理）"""
    CSV_PATH = Path(config['system_config']['root_dir']) / 'data' / 'economic_calendar.csv'
    start_date = pd.to_datetime(date_range['start']) - timedelta(days=7)
    end_date = pd.to_datetime(date_range['end']) + timedelta(days=7)
    try:
        # 使用非同步 SQLAlchemy 連線
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
        query = text("SELECT * FROM economic_calendar WHERE date BETWEEN :start_date AND :end_date")
        async with engine.connect() as conn:
            result = await conn.execute(query, {'start_date': start_date.strftime('%Y-%m-%d'), 'end_date': end_date.strftime('%Y-%m-%d')})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            df['date'] = pd.to_datetime(df['date'])
            if not df.empty:
                df = filter_future_dates(df)
                df = fill_missing_values(df)
                logging.info(f"Loaded economic calendar from SQLite: shape={df.shape}")
                return df[['date', 'event', 'impact', 'fed_funds_rate']]
    except Exception as e:
        logging.error(f"SQLite economic calendar query failed: {str(e)}, traceback={traceback.format_exc()}")
    try:
        importances = ['high', 'medium']
        time_zone = 'GMT +8:00'
        from_date = start_date.strftime('%d/%m/%Y')
        to_date = end_date.strftime('%d/%m/%Y')
        logging.info(f"Fetching economic calendar from investpy: start={from_date}, end={to_date}")
        calendar = investpy.economic_calendar(importances=importances, time_zone=time_zone, from_date=from_date, to_date=to_date, countries=['united states', 'japan'])
        if calendar.empty:
            logging.warning("investpy economic calendar data is empty")
            df = pd.DataFrame()
        else:
            calendar['date'] = pd.to_datetime(calendar['date'], format='%d/%m/%Y')
            calendar = calendar[calendar['importance'].notnull()]
            calendar['event'] = calendar['currency'] + ' ' + calendar['event']
            calendar['impact'] = calendar['importance'].str.capitalize()
            df = calendar[['date', 'event', 'impact']]
    except Exception as e:
        logging.error(f"Failed to fetch investpy economic calendar: {str(e)}, traceback={traceback.format_exc()}")
        df = pd.DataFrame()
    try:
        fred_api_key = config['api_key'].get('fred_api_key', '')
        if not fred_api_key:
            logging.error("FRED API key not configured")
        else:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={fred_api_key}&file_type=json&observation_start={start_date.strftime('%Y-%m-%d')}&observation_end={end_date.strftime('%Y-%m-%d')}"
            async with aiohttp.ClientSession() as session:
                async with (await make_get_request(session, url, timeout=10)) as response:
                    data = await response.json()
                    if 'observations' not in data:
                        logging.warning("FRED API returned no observations")
                    else:
                        fred_data = pd.DataFrame(data['observations'])[['date', 'value']].rename(columns={'value': 'fed_funds_rate'})
                        fred_data['date'] = pd.to_datetime(fred_data['date'])
                        fred_data['fed_funds_rate'] = fred_data['fed_funds_rate'].astype(float)
                        if not df.empty:
                            df = df.merge(fred_data[['date', 'fed_funds_rate']], on='date', how='left')
                        else:
                            df = fred_data[['date', 'fed_funds_rate']]
                            df['event'] = 'FEDFUNDS'
                            df['impact'] = 'High'
    except Exception as e:
        logging.error(f"Failed to fetch FRED API data: {str(e)}, traceback={traceback.format_exc()}")
    if not df.empty:
        df = df.drop_duplicates(subset=['date', 'event'], keep='last')
        df = filter_future_dates(df)
        df = fill_missing_values(df)
        # 修改：確保 'fed_funds_rate' 存在並填充
        if 'fed_funds_rate' not in df.columns:
            df['fed_funds_rate'] = np.nan  # 讓 DB default 處理
            logging.warning("economic_calendar 缺少 'fed_funds_rate'，依賴 DB default")
        await save_data(df, timeframe='1 day', db_path=db_path, data_type='economic')
        logging.info(f"Economic calendar data saved to SQLite: shape={df.shape}")
        if not os.path.exists(CSV_PATH.parent):
            os.makedirs(CSV_PATH.parent)
        df.to_csv(CSV_PATH, index=False)
        logging.info(f"Economic calendar data saved to CSV: {CSV_PATH}")
        return df[['date', 'event', 'impact', 'fed_funds_rate']]
    logging.info("No investpy/FRED data, trying to load from CSV")
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        required_columns = ['date', 'event', 'impact']
        if all(col in df.columns for col in required_columns):
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            df = df.drop_duplicates(subset=['date', 'event'], keep='last')
            df = filter_future_dates(df)
            df = fill_missing_values(df)
            if 'fed_funds_rate' not in df.columns:
                df['fed_funds_rate'] = np.nan  # 讓 DB default 處理
                logging.warning("CSV 缺少 'fed_funds_rate'，依賴 DB default")
            await save_data(df, timeframe='1 day', db_path=db_path, data_type='economic')
            logging.info(f"Loaded and saved economic calendar from CSV: shape={df.shape}")
            return df[['date', 'event', 'impact', 'fed_funds_rate']]
        else:
            logging.warning(f"Invalid or missing columns in {CSV_PATH}, columns={df.columns.tolist()}")
    logging.warning("Economic calendar is empty")
    return pd.DataFrame()
```

## main.py

```python
import asyncio
import logging
import logging.handlers
import pandas as pd
import torch
import json
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from data_acquisition import fetch_data, compute_indicators, fetch_economic_calendar
from ai_models import update_model, predict_sentiment, integrate_sentiment
from trading_strategy import ForexEnv, train_ppo, make_decision, backtest, connect_ib, execute_trade
from risk_management import calculate_stop_loss, calculate_take_profit, calculate_position_size, predict_volatility
from utils import check_hardware, test_proxy, get_proxy, check_volatility, save_periodically, initialize_db, load_settings, decrypt_key, check_table_data
import streamlit as st
from prometheus_client import Counter, Histogram
from pathlib import Path
import numpy as np

# Prometheus 指標，用於監控交易次數和 API 延遲
trade_counter = Counter('usd_jpy_trades_total', 'Total number of trades executed', ['action', 'mode'])
api_latency = Histogram('usd_jpy_api_latency_seconds', 'API call latency', ['mode'])

# 結構化 JSON 日誌格式，方便後續分析
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'filename': record.filename,
            'funcName': record.funcName,
            'mode': getattr(record, 'mode', 'unknown')
        }
        return json.dumps(log_data, ensure_ascii=False)

def setup_logging(mode: str):
    """設置日誌：根據模式創建日誌檔案，使用 JSON 格式。（優化：合併 logging 設置）"""
    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'app_{mode}_{log_time}.log'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.handlers.TimedRotatingFileHandler(log_file, when='midnight', backupCount=7, encoding='utf-8')
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logging.info(f"Logging setup completed for mode: {mode}", extra={'mode': mode})

def clean_old_backups(root_dir: str, days_to_keep: int = 7):
    """清理舊備份檔案：僅保留指定天數的備份。（優化：使用 list comprehension 簡化遍歷）"""
    backup_dir = Path(root_dir) / 'backups'
    if not backup_dir.exists():
        return
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    [backup_file.unlink() for backup_file in backup_dir.glob('trading_data_*.db') if datetime.strptime(backup_file.stem.split('_')[-1], '%Y%m%d') < cutoff_date]
    logging.info("Old backup files cleaned", extra={'mode': 'cleanup'})

def load_config():
    """載入配置檔：從環境變數和 JSON 載入。（優化：合併解密邏輯）"""
    load_dotenv()
    config = load_settings()
    if config is None or not isinstance(config, dict):
        logging.error("load_settings() 返回 None 或非字典，使用預設配置")
        config = {}
    api_key = config.get('api_key', {})
    if api_key is None:
        logging.error("api_key 是 None，使用預設空字典")
        api_key = {}
    api_key = {k: decrypt_key(v) if isinstance(v, bytes) else v for k, v in api_key.items()}
    system_config = config.get('system_config', {})
    trading_params = config.get('trading_params', {})
    fernet_key = api_key.get('FERNET_KEY', '')
    if not fernet_key:
        fernet_key = "default_fernet_key"  # 預設值，避免 raise
        logging.warning("FERNET_KEY 未設置，使用預設值")
    return config, api_key, system_config, trading_params

async def main(mode: str = 'backtest'):
    """主程式入口：協調所有功能。（優化：合併 logging/print，簡化任務列表）"""
    setup_logging(mode)
    logging.info(f"Starting program in {mode} mode", extra={'mode': mode})
    try:
        config, api_key, system_config, trading_params = load_config()
        logging.info("Configuration loaded successfully", extra={'mode': mode})
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}", extra={'mode': mode})
        return
    if not await test_proxy(await get_proxy(load_settings())) and not config['system_config'].get('offline_mode', False):
        logging.error("無法連網且未啟用離線模式，程式退出")
        exit(1)
    device_config, onnx_session = check_hardware()
    db_path = system_config['db_path']
    await initialize_db(db_path)
    logging.info("Database initialized", extra={'mode': mode})
    # 添加資料表數據檢查
    tables_to_check = ['tweets', 'sentiment_data', 'trades']
    for table in tables_to_check:
        has_data = await check_table_data(db_path, table)
        if not has_data:
            logging.warning(f"{table} 表無數據，可能導致訓練或分析錯誤")
    date_range = {
        'start': (datetime.now() - pd.Timedelta(days=trading_params['min_backtest_days'])).strftime('%Y-%m-%d'),
        'end': datetime.now().strftime('%Y-%m-%d')
    }
    timeframes = ['1h', '4h', '1d']
    data_frames = {}
    tasks = []
    for tf in timeframes:
        start_time = time.time()
        df = await fetch_data(system_config['data_source'], ['yfinance'], date_range, tf, db_path, config)
        api_latency.labels(mode=mode).observe(time.time() - start_time)
        if df.empty:
            logging.error(f"Failed to fetch {tf} data", extra={'mode': mode})
            [task.cancel() for task in tasks]
            return
        df = await compute_indicators(df, db_path, tf, config)
        data_frames[tf] = df
        logging.info(f"{tf} data preprocessing completed", extra={'mode': mode})
    common_dates = set.intersection(*(set(df['date']) for df in data_frames.values() if not df.empty))
    for tf in timeframes:
        if not data_frames[tf].empty:
            data_frames[tf] = data_frames[tf][data_frames[tf]['date'].isin(common_dates)].copy()
            logging.info(f"Aligned {tf} data to common dates, rows={len(data_frames[tf])}")
    economic_calendar = await fetch_economic_calendar(date_range, db_path, config)
    if not economic_calendar.empty:
        logging.info("Economic calendar is not empty", extra={'mode': mode})
        for tf in timeframes:
            data_frames[tf] = data_frames[tf].merge(
                economic_calendar[['date', 'event', 'impact', 'fed_funds_rate']], on='date', how='left'
            )
            data_frames[tf]['impact'] = data_frames[tf]['impact'].fillna('Low')
            # 修改：檢查 'fed_funds_rate' 是否存在，避免 KeyError
            if 'fed_funds_rate' in data_frames[tf].columns:
                data_frames[tf]['fed_funds_rate'] = data_frames[tf]['fed_funds_rate'].ffill().bfill()
            else:
                data_frames[tf]['fed_funds_rate'] = np.nan  # 讓 DB default 處理
                logging.warning(f"{tf} data 缺少 'fed_funds_rate' 欄位，依賴 DB default")
    # 修改：只在 LIVE 模式下創建定期保存 tasks
    if mode == 'live':
        tasks = [asyncio.create_task(save_periodically(data_frames[tf], tf, db_path, system_config['root_dir'], data_type=dtype)) for tf in timeframes for dtype in ('ohlc',) if not data_frames[tf].empty] + [asyncio.create_task(save_periodically(economic_calendar, '1d', db_path, system_config['root_dir'], 'economic'))] if not economic_calendar.empty else []
    clean_old_backups(system_config['root_dir'])
    logging.info("Old backup files cleaned", extra={'mode': mode})
    session = 'normal'
    if '1h' in data_frames and not data_frames['1h'].empty and 'ATR' in data_frames['1h'].columns:
        volatility_level = check_volatility(data_frames['1h']['ATR'].mean())
        if volatility_level == 'high':
            logging.warning("High volatility detected, entering conservative mode", extra={'mode': mode})
            trading_params['risk_percent'] *= 0.5
            session = 'high_volatility'
    models = update_model(data_frames['1d'], 'models', session, device_config)
    if not models:
        logging.error("Model update failed", extra={'mode': mode})
        [task.cancel() for task in tasks]
        return
    sentiment_score = await predict_sentiment(date_range['end'], db_path, config)
    sentiment_adjustment = integrate_sentiment(sentiment_score)
    logging.info(f"Sentiment analysis result: score={sentiment_score}, adjustment={sentiment_adjustment}", extra={'mode': mode})
    env = ForexEnv(data_frames)
    ppo_model = train_ppo(env, device_config)
    logging.info("PPO model training completed", extra={'mode': mode})
    action = make_decision(ppo_model, data_frames, sentiment_score)
    logging.info(f"Trading decision: {action}", extra={'mode': mode})
    if '1h' not in data_frames or data_frames['1h'].empty or 'ATR' in data_frames['1h'].columns or 'close' in data_frames['1h'].columns:
        logging.error("1h data or required columns missing, cannot proceed with risk management", extra={'mode': mode})
        [task.cancel() for task in tasks]
        return
    atr = data_frames['1h']['ATR'].iloc[-1]
    predicted_vol = predict_volatility(data_frames['1h'], 'models/lightgbm_model_quantized.onnx')
    current_price = data_frames['1h']['close'].iloc[-1]
    stop_loss = calculate_stop_loss(current_price, atr, action)
    take_profit = calculate_take_profit(current_price, atr, action)
    position_size = await calculate_position_size(trading_params['capital'], trading_params['risk_percent'], abs(current_price - stop_loss), sentiment_score, db_path, volatility_level)
    logging.info(f"Action: {action}, Stop Loss: {stop_loss}, Take Profit: {take_profit}, Position Size: {position_size}", extra={'mode': mode})
    trade = {'action': action, 'price': current_price, 'quantity': position_size, 'stop_loss': stop_loss, 'take_profit': take_profit, 'leverage': 1}
    if not compliance_check(trade):
        logging.warning("Trade does not comply with leverage limits", extra={'mode': mode})
        [task.cancel() for task in tasks]
        return
    if mode == 'backtest':
        async def async_strategy(row):
            return make_decision(ppo_model, data_frames, sentiment_score)
        result = await backtest(data_frames['1d'], async_strategy, trading_params['capital'])
        logging.info(f"Backtest Results: {result}", extra={'mode': 'backtest'})
        report_dir = Path('reports')
        report_dir.mkdir(exist_ok=True)
        pd.DataFrame([result]).to_csv(report_dir / f'backtest_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
        logging.info("Backtest completed, report generated", extra={'mode': 'backtest'})
    elif mode == 'live':
        ib = connect_ib()
        trade_counter.labels(action=action, mode=mode).inc()
        execute_trade(ib, action, current_price, position_size, stop_loss, take_profit)
        st.title("USD/JPY 交易儀表板")
        st.line_chart(data_frames['1h']['close'])
        st.write("最新決策:", action)
        st.write("最新指標:", data_frames['1h'][['RSI', 'MACD', 'Stoch_k', 'ADX', 'BB_upper', 'BB_lower', 'EMA_12', 'EMA_26']].iloc[-1])
        override_action = st.selectbox("手動覆寫決策", ["無", "買入", "賣出", "持有"])
        if override_action != "無":
            action = override_action
            logging.info(f"User overridden decision: {action}", extra={'mode': mode})
            trade_counter.labels(action=action, mode=mode).inc()
            execute_trade(ib, action, current_price, position_size, stop_loss, take_profit)
        logging.info("Live trading mode running", extra={'mode': mode})
    [task.cancel() for task in tasks]
    logging.info("Program execution completed", extra={'mode': mode})

def compliance_check(trade: dict) -> bool:
    """檢查交易是否符合槓桿限制。"""
    leverage = trade.get('leverage', 1)
    is_compliant = leverage <= 30
    logging.info(f"Leverage check: leverage={leverage}, compliant={is_compliant}", extra={'mode': 'compliance'})
    return is_compliant

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="USD/JPY 自動交易系統")
    parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest', help="運行模式：回測或實時交易")
    args = parser.parse_args()
    asyncio.run(main(args.mode))
```

## risk_management.py

```python
import pandas as pd
import logging
from psutil import virtual_memory, cpu_percent
import onnxruntime as ort
import numpy as np
from pathlib import Path
import aiosqlite
from ai_models import FEATURES
import traceback

async def get_current_exposure(db_path: str) -> float:
    """獲取當前總持倉暴露。"""
    try:
        async with aiosqlite.connect(db_path, timeout=10) as conn:
            cursor = await conn.execute("SELECT SUM(volume * price) as total_exposure FROM trades WHERE action IN ('買入', '賣出')")
            result = await cursor.fetchone()
            total_exposure = result[0] or 0.0
            logging.info(f"當前總持倉暴露: {total_exposure}")
            return total_exposure
    except Exception as e:
        logging.error(f"獲取持倉暴露失敗: {e}, traceback={traceback.format_exc()}")
        return 0.0

def calculate_stop_loss(current_price: float, atr: float, action: str, multiplier: float = 2) -> float:
    """計算止損。"""
    try:
        if action == "買入":
            stop_loss = current_price - (multiplier * atr)
        elif action == "賣出":
            stop_loss = current_price + (multiplier * atr)
        else:
            stop_loss = current_price
        logging.info(f"計算止損: 當前價格={current_price}, ATR={atr}, 行動={action}, 止損={stop_loss}")
        return stop_loss
    except Exception as e:
        logging.error(f"止損計算錯誤: {e}, traceback={traceback.format_exc()}")
        return current_price

def calculate_take_profit(current_price: float, atr: float, action: str, multiplier: float = 2) -> float:
    """計算止盈。"""
    try:
        if action == "買入":
            take_profit = current_price + (multiplier * atr)
        elif action == "賣出":
            take_profit = current_price - (multiplier * atr)
        else:
            take_profit = current_price
        logging.info(f"計算止盈: 當前價格={current_price}, ATR={atr}, 行動={action}, 止盈={take_profit}")
        return take_profit
    except Exception as e:
        logging.error(f"止盈計算錯誤: {e}, traceback={traceback.format_exc()}")
        return current_price

def check_resources(threshold_mem: float = 0.9, threshold_cpu: float = 80.0) -> bool:
    """檢查系統資源。"""
    try:
        mem = virtual_memory()
        cpu = cpu_percent(interval=1)
        if mem.percent > threshold_mem * 100 or cpu > threshold_cpu:
            logging.warning(f"資源使用過高：記憶體 {mem.percent}%，CPU {cpu}%")
            return False
        logging.info(f"資源檢查通過：記憶體 {mem.percent}%，CPU {cpu}%")
        return True
    except Exception as e:
        logging.error(f"資源檢查錯誤: {e}, traceback={traceback.format_exc()}")
        return False

async def calculate_position_size(capital: float, risk_percent: float, stop_loss_distance: float, sentiment: float = 0.0, db_path: str = "C:\\Trading\\data\\trading_data.db", volatility_level: str = 'normal') -> float:
    """計算倉位大小。"""
    try:
        if abs(sentiment) > 0.8:
            logging.warning(f"極端情緒分數: {sentiment}，倉位大小設為 0")
            return 0.0
        base_size = (capital * risk_percent) / stop_loss_distance if stop_loss_distance > 0 else 0
        adjustment = 1.2 if sentiment > 0.4 else 0.8 if sentiment < -0.4 else 1.0
        position_size = base_size * adjustment * (0.5 if volatility_level == 'high' else 1.0)
        if volatility_level == 'high':
            logging.info("高波動模式，倉位減半")
        total_exposure = await get_current_exposure(db_path)
        max_exposure = capital * 0.05
        if total_exposure + (position_size * stop_loss_distance) > max_exposure:
            logging.warning(f"超過最大暴露限額: 當前={total_exposure}, 擬新增={position_size * stop_loss_distance}, 限額={max_exposure}")
            return 0.0
        leverage = (position_size * stop_loss_distance) / capital if capital > 0 else 0
        if leverage > 30:
            logging.warning(f"槓桿超過 30:1: 計算槓桿={leverage:.2f}")
            return 0.0
        logging.info(f"計算倉位大小：基礎={base_size:.2f}，情緒調整={adjustment:.2f}，最終={position_size:.2f}，槓桿={leverage:.2f}")
        return position_size
    except Exception as e:
        logging.error(f"倉位計算錯誤: {e}")
        return 0

def predict_volatility(df: pd.DataFrame, model_path: str = 'models/lightgbm_model_quantized.onnx') -> float:
    """預測波動性。"""
    try:
        for f in FEATURES:
            if f not in df.columns:
                df[f] = 0
        X = df[FEATURES].iloc[-1:].values.astype(np.float16)  # 修改為 float16
        if len(X) == 0:
            logging.error("X 數據為空，回退到平均 ATR")
            return df['ATR'].mean()
        Path(model_path).parent.mkdir(exist_ok=True)
        session = ort.InferenceSession(model_path, providers=['VitisAIExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        pred = session.run(None, {'input': X})[0][0]
        logging.info(f"LightGBM 波動性預測: {pred}")
        return pred
    except Exception as e:
        logging.error(f"波動預測錯誤: {e}")
        return df['ATR'].mean()
```

## trading_strategy.py

```python
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import pandas as pd
from ib_insync import IB, Forex, BracketOrder
import logging
from risk_management import check_resources, calculate_stop_loss, calculate_position_size
from utils import load_settings
from ai_models import FEATURES
import torch
import traceback
import asyncio
import json

class ForexEnv(gym.Env):
    """外匯環境：用於 PPO 強化學習訓練。"""
    def __init__(self, data_frames: dict, spread: float = 0.0002):
        super().__init__()
        tf_mapping = {'1 hour': '1h', '4 hours': '4h', '1 day': '1d'}
        self.data_frames = {tf_mapping.get(k, k): v for k, v in data_frames.items()}
        self.spread = spread
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)
        # 確保觀察空間與特徵數量匹配
        feature_count = len(FEATURES) - 1  # 排除 fed_funds_rate
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(feature_count * 3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for tf in ['1h', '4h', '1d']:
            df = self.data_frames.get(tf, pd.DataFrame())
            if not df.empty and self.current_step < len(df) and all(f in df.columns for f in FEATURES[:-1]):
                obs.append(df[FEATURES[:-1]].iloc[self.current_step].values)
            else:
                obs.append(np.zeros(len(FEATURES) - 1))
        return np.array(obs, dtype=np.float32).flatten()

    def step(self, action):
        price = self.data_frames['1h']['close'].iloc[self.current_step] if self.current_step < len(self.data_frames['1h']) else 0.0
        reward = -self.spread if action in [0, 1] else 0
        self.current_step += 1
        done = self.current_step >= min(len(self.data_frames[tf]) for tf in self.data_frames if not self.data_frames[tf].empty) - 1
        return self._get_obs(), reward, done, False, {}

def train_ppo(env, device_config: dict = None):
    """訓練 PPO。"""
    try:
        config = load_settings()
        total_timesteps = config.get('trading_params', {}).get('ppo_timesteps', 1000)
        learning_rate = config.get('trading_params', {}).get('ppo_learning_rate', 0.0003)
        device = 'cpu'
        logging.info(f"PPO 訓練：使用 total_timesteps={total_timesteps}, learning_rate={learning_rate}, device={device}")
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=learning_rate, device=device)
        model.learn(total_timesteps=total_timesteps)
        model.save("models/ppo_model")
        logging.info("PPO 模型訓練完成")
        return model
    except Exception as e:
        logging.error(f"PPO 訓練錯誤: {e}, traceback={traceback.format_exc()}")
        return None

def make_decision(model, data_frames: dict, sentiment: float) -> str:
    """產生決策，並記錄詳細原因。"""
    try:
        if not check_resources():
            logging.warning("資源不足，決策為持有", extra={'mode': 'decision'})
            print("決策：持有，原因：系統資源不足")
            return "持有"
        if model is None:
            logging.warning("PPO 模型為 None，決策為持有", extra={'mode': 'decision'})
            print("決策：持有，原因：PPO 模型未正確加載")
            return "持有"
        
        buy_signals = []
        sell_signals = []
        decision_log = {"buy_signals": {}, "sell_signals": {}, "ppo_action": "", "sentiment_adjust": 0.0}
        
        for tf in ['1h', '4h', '1d']:
            df = data_frames.get(tf, pd.DataFrame())
            if df.empty or not all(f in df.columns for f in ['RSI', 'MACD', 'MACD_signal', 'Stoch_k', 'ADX', 'Ichimoku_tenkan', 'Ichimoku_kijun', 'Ichimoku_cloud_top', 'BB_lower', 'BB_upper', 'EMA_12', 'EMA_26']):
                logging.warning(f"{tf} 數據為空或缺少必要欄位，跳過該時間框架", extra={'mode': 'decision'})
                decision_log[tf] = {"status": "skipped", "reason": "數據缺失或欄位不全"}
                continue
            
            rsi, macd, macd_signal, stoch_k, adx = df[['RSI', 'MACD', 'MACD_signal', 'Stoch_k', 'ADX']].iloc[-1]
            ichimoku_buy = (df['Ichimoku_tenkan'].iloc[-1] > df['Ichimoku_kijun'].iloc[-1] and df['close'].iloc[-1] > df['Ichimoku_cloud_top'].iloc[-1])
            ichimoku_sell = (df['Ichimoku_tenkan'].iloc[-1] < df['Ichimoku_kijun'].iloc[-1] and df['close'].iloc[-1] < df['Ichimoku_cloud_top'].iloc[-1])
            bb_buy = df['close'].iloc[-1] < df['BB_lower'].iloc[-1]
            bb_sell = df['close'].iloc[-1] > df['BB_upper'].iloc[-1]
            ema_signal = df['EMA_12'].iloc[-1] > df['EMA_26'].iloc[-1]
            economic_impact = df['impact'].iloc[-1] if 'impact' in df.columns and not pd.isna(df['impact'].iloc[-1]) else 'Low'
            economic_pause = economic_impact in ['High', 'Medium']
            
            buy_condition = (rsi < 30 and macd > macd_signal and stoch_k < 20 and adx > 25 and ichimoku_buy and bb_buy and ema_signal and not economic_pause)
            sell_condition = (rsi > 70 and macd < macd_signal and stoch_k > 80 and adx > 25 and ichimoku_sell and bb_sell and df['EMA_12'].iloc[-1] < df['EMA_26'].iloc[-1] and not economic_pause)
            
            buy_signals.append(buy_condition)
            sell_signals.append(sell_condition)
            
            decision_log[tf] = {
                "rsi": float(rsi),
                "macd": float(macd),
                "macd_signal": float(macd_signal),
                "stoch_k": float(stoch_k),
                "adx": float(adx),
                "ichimoku_buy": bool(ichimoku_buy),  # 將 numpy.bool_ 轉為 Python bool
                "ichimoku_sell": bool(ichimoku_sell),  # 將 numpy.bool_ 轉為 Python bool
                "bb_buy": bool(bb_buy),  # 將 numpy.bool_ 轉為 Python bool
                "bb_sell": bool(bb_sell),  # 將 numpy.bool_ 轉為 Python bool
                "ema_signal": bool(ema_signal),  # 將 numpy.bool_ 轉為 Python bool
                "economic_impact": economic_impact,
                "buy_condition": bool(buy_condition),  # 將 numpy.bool_ 轉為 Python bool
                "sell_condition": bool(sell_condition)  # 將 numpy.bool_ 轉為 Python bool
            }
        
        buy_score = sum(buy_signals) / len(buy_signals) if buy_signals else 0
        sell_score = sum(sell_signals) / len(sell_signals) if sell_signals else 0
        
        if abs(sentiment) > 0.8:
            logging.warning(f"極端情緒分數: {sentiment}，決策為持有", extra={'mode': 'decision'})
            print(f"決策：持有，原因：情緒分數 {sentiment:.2f} 過高")
            decision_log["reason"] = f"極端情緒分數: {sentiment}"
            logging.info(f"決策詳情: {json.dumps(decision_log, ensure_ascii=False)}", extra={'mode': 'decision'})
            return "持有"
        
        sentiment_adjust = 0.2 if sentiment > 0.4 else -0.2 if sentiment < -0.4 else 0.0
        buy_score += sentiment_adjust
        sell_score -= sentiment_adjust
        decision_log["sentiment_adjust"] = sentiment_adjust
        
        obs = []
        for tf in ['1h', '4h', '1d']:
            df = data_frames.get(tf, pd.DataFrame())
            if not df.empty and all(f in df.columns for f in FEATURES[:-1]):
                obs.append(df[FEATURES[:-1]].iloc[-1].values)
            else:
                obs.append(np.zeros(len(FEATURES) - 1))
        obs = np.array(obs, dtype=np.float32).flatten()
        action, _ = model.predict(obs)
        ppo_action = ["買入", "賣出", "持有"][action]
        decision_log["ppo_action"] = ppo_action
        
        if buy_score > 0.6 and ppo_action == "買入":
            decision_log["final_decision"] = "買入"
            logging.info(f"決策：買入，買入得分={buy_score:.2f}, PPO行動={ppo_action}, 情緒調整={sentiment_adjust:.2f}", extra={'mode': 'decision'})
            print(f"決策：買入，原因：買入得分 {buy_score:.2f}，PPO 預測買入，情緒調整 {sentiment_adjust:.2f}")
            logging.info(f"決策詳情: {json.dumps(decision_log, ensure_ascii=False)}", extra={'mode': 'decision'})
            return "買入"
        elif sell_score > 0.6 and ppo_action == "賣出":
            decision_log["final_decision"] = "賣出"
            logging.info(f"決策：賣出，賣出得分={sell_score:.2f}, PPO行動={ppo_action}, 情緒調整={sentiment_adjust:.2f}", extra={'mode': 'decision'})
            print(f"決策：賣出，原因：賣出得分 {sell_score:.2f}，PPO 預測賣出，情緒調整 {sentiment_adjust:.2f}")
            logging.info(f"決策詳情: {json.dumps(decision_log, ensure_ascii=False)}", extra={'mode': 'decision'})
            return "賣出"
        else:
            decision_log["final_decision"] = "持有"
            reason = f"買入得分 {buy_score:.2f} 或賣出得分 {sell_score:.2f} 未達門檻，或 PPO 行動 {ppo_action} 不匹配"
            logging.info(f"決策：持有，原因：{reason}", extra={'mode': 'decision'})
            print(f"決策：持有，原因：{reason}")
            logging.info(f"決策詳情: {json.dumps(decision_log, ensure_ascii=False)}", extra={'mode': 'decision'})
            return "持有"
    except Exception as e:
        logging.error(f"決策錯誤: {e}, traceback={traceback.format_exc()}", extra={'mode': 'decision'})
        print(f"決策：持有，原因：決策過程發生錯誤 {str(e)}")
        return "持有"

async def backtest(df: pd.DataFrame, strategy: callable, initial_capital: float = 10000, spread: float = 0.0002) -> dict:
    """回測：模擬交易，顯示每日進度。"""
    capital = initial_capital
    position_size = 0.0
    entry_price = None
    trades = []
    equity_curve = []
    total_steps = len(df)
    last_date = None
    
    for i, row in df.iterrows():
        current_date = row['date'].date()
        if last_date != current_date:
            progress = (i + 1) / total_steps * 100
            logging.info(f"回測進度：日期={current_date}, 進度={progress:.2f}%", extra={'mode': 'backtest'})
            print(f"回測進度：日期={current_date}, 進度={progress:.2f}%")
            last_date = current_date
        
        if not check_resources():
            logging.warning("資源不足，跳過交易", extra={'mode': 'backtest'})
            continue
        
        action = await strategy(row)
        current_price = row['close']
        sentiment = row.get('sentiment', 0.0)
        atr = row['ATR']
        stop_loss_distance = abs(calculate_stop_loss(current_price, atr, action) - current_price)
        
        # 使用事件循環運行異步函數
        # 異步計算倉位大小
        position_size_calc = await calculate_position_size(initial_capital, 0.01, stop_loss_distance, sentiment)
        
        if action == "買入" and position_size <= 0:
            if position_size < 0:
                profit = (entry_price - current_price) * abs(position_size)
                capital += profit
                trades.append({'date': row['date'], 'action': '平空', 'price': current_price, 'profit': profit, 'position_size': position_size})
            position_size = position_size_calc
            entry_price = current_price
            capital -= abs(position_size) * 0.0001
            trades.append({'date': row['date'], 'action': '買入', 'price': current_price, 'profit': 0.0, 'position_size': position_size})
        elif action == "賣出" and position_size >= 0:
            if position_size > 0:
                profit = (current_price - entry_price) * position_size
                capital += profit
                trades.append({'date': row['date'], 'action': '平多', 'price': current_price, 'profit': profit, 'position_size': position_size})
            position_size = -position_size_calc
            entry_price = current_price
            capital -= abs(position_size) * 0.0001
            trades.append({'date': row['date'], 'action': '賣出', 'price': current_price, 'profit': 0.0, 'position_size': position_size})
        
        equity = capital + (current_price - entry_price) * position_size if position_size != 0 else capital
        equity_curve.append(equity)
    
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()
    result = {
        "final_capital": capital,
        "sharpe_ratio": (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
        "max_drawdown": (equity_series / equity_series.cummax() - 1).min(),
        "win_rate": len([r for r in returns if r > 0]) / len(returns) if len(returns) > 0 else 0,
        "trades": trades
    }
    logging.info(f"回測結果：最終資本={capital:.2f}, 夏普比率={result['sharpe_ratio']:.2f}, 最大回撤={result['max_drawdown']:.2f}, 勝率={result['win_rate']:.2f}", extra={'mode': 'backtest'})
    return result

def connect_ib(host='127.0.0.1', port=7497, client_id=1):
    """連接 IB API。"""
    ib = IB()
    ib.connect(host, port, client_id)
    return ib

def execute_trade(ib, action: str, price: float, quantity: float, stop_loss: float, take_profit: float):
    """執行交易。"""
    contract = Forex('USDJPY')
    for attempt in range(3):
        try:
            if action == "買入":
                order = BracketOrder('BUY', quantity, price, takeProfitPrice=take_profit, stopLossPrice=stop_loss)
            elif action == "賣出":
                order = BracketOrder('SELL', quantity, price, takeProfitPrice=stop_loss, stopLossPrice=take_profit)
            else:
                return
            trade = ib.placeOrder(contract, order)
            ib.sleep(1)
            if trade.orderStatus.status in ['Filled', 'Submitted']:
                logging.info(f"訂單狀態: {trade.orderStatus.status}")
                return
            else:
                logging.warning(f"訂單失敗: {trade.orderStatus.status}, 重試 {attempt + 1}/3")
        except Exception as e:
            logging.error(f"交易執行錯誤: {e}, 重試 {attempt + 1}/3")
        ib.sleep(2 ** attempt * 2)
    logging.error("交易執行失敗，超過最大重試次數")
```

## utils.py

```python
import torch
import onnxruntime as ort
import aiohttp
import logging
import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from pathlib import Path
from datetime import datetime
import json
import traceback
import aiofiles
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text, insert, Table, MetaData, Column, DateTime, Float, Integer, Text
import pandas as pd
import time #response.get要有time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [%(module)s]',
    handlers=[
        logging.FileHandler('C:/Trading/logs/app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

_config_cache = None
_proxy_cache = None
_proxy_tested = None

key = b'_eIKG0YhiJCyBQ-VvxAsx8LT3Vow-k0hE-i0iwK9wwM='
cipher = Fernet(key)

def encrypt_key(api_key: str) -> bytes:
    return cipher.encrypt(api_key.encode())

def decrypt_key(encrypted: bytes) -> str:
    return cipher.decrypt(encrypted).decode()

async def fetch_api_data(url: str, headers: dict = None, params: dict = None, proxies: dict = None) -> dict:
    """通用 API 獲取。"""
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                response = await make_get_request(session, url, headers=headers, params=params, proxies=proxies, timeout=10)
                data = await response.json()
                return data
            except Exception as e:
                if attempt == 2:
                    logging.error(f"API 獲取失敗: {str(e)}, URL={url}")
                    return {}
                await asyncio.sleep(2 ** attempt * 2)
    return {}

async def initialize_db(db_path: str):
    """初始化資料庫，使用非同步SQLAlchemy。"""
    config = load_settings()
    root_dir = config.get('system_config', {}).get('root_dir', str(Path(db_path).parent))
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        try:
            engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1 FROM sqlite_master"))
        except Exception:
            logging.warning(f"資料庫檔案 {db_path} 損壞，將刪除並重新創建")
            os.remove(db_path)
    try:
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", connect_args={"timeout": 30}, isolation_level="SERIALIZABLE")
        async with engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ohlc (
                    date DATETIME,
                    open REAL DEFAULT 0.0,
                    high REAL DEFAULT 0.0,
                    low REAL DEFAULT 0.0,
                    close REAL DEFAULT 0.0,
                    volume INTEGER DEFAULT 0,
                    n INTEGER DEFAULT 0,
                    vw REAL DEFAULT 0.0,
                    timeframe TEXT,
                    RSI REAL DEFAULT 0.0,
                    MACD REAL DEFAULT 0.0,
                    MACD_signal REAL DEFAULT 0.0,
                    MACD_hist REAL DEFAULT 0.0,
                    ATR REAL DEFAULT 0.0,
                    Stoch_k REAL DEFAULT 0.0,
                    ADX REAL DEFAULT 0.0,
                    Ichimoku_tenkan REAL DEFAULT 0.0,
                    Ichimoku_kijun REAL DEFAULT 0.0,
                    Ichimoku_span_a REAL DEFAULT 0.0,
                    Ichimoku_span_b REAL DEFAULT 0.0,
                    Ichimoku_cloud_top REAL DEFAULT 0.0,
                    BB_upper REAL DEFAULT 0.0,
                    BB_middle REAL DEFAULT 0.0,
                    BB_lower REAL DEFAULT 0.0,
                    EMA_12 REAL DEFAULT 0.0,
                    EMA_26 REAL DEFAULT 0.0,
                    fed_funds_rate REAL DEFAULT 0.0,
                    PRIMARY KEY (date, timeframe)
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS economic_calendar (
                    date DATETIME,
                    event TEXT,
                    impact TEXT,
                    fed_funds_rate REAL DEFAULT 0.0,
                    PRIMARY KEY (date, event)
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    date DATETIME,
                    sentiment REAL DEFAULT 0.0,
                    PRIMARY KEY (date)
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tweets (
                    date DATETIME,
                    tweet_id TEXT,
                    text TEXT,
                    user_id TEXT,
                    PRIMARY KEY (date, tweet_id)
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT,
                    verified BOOLEAN DEFAULT FALSE
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    price REAL DEFAULT 0.0,
                    action TEXT,
                    volume REAL DEFAULT 0.0,
                    stop_loss REAL DEFAULT 0.0,
                    take_profit REAL DEFAULT 0.0
                )
            """))
            # 添加索引優化
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_date ON ohlc(date)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_date_event ON economic_calendar(date, event)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_date_tweet ON tweets(date, tweet_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sentiment_date ON sentiment_data(date)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_id ON users(user_id)"))
        await backup_database(db_path, root_dir)
        logging.info(f"資料庫初始化完成: {db_path}")
    except Exception as e:
        logging.error(f"資料庫初始化失敗: {str(e)}, traceback={traceback.format_exc()}")
        raise

async def save_data(df: pd.DataFrame, timeframe: str, db_path: str, data_type: str = 'ohlc') -> bool:
    """增量儲存到 SQLite，使用非同步SQLAlchemy，明確定義表格結構。"""
    for attempt in range(3):
        try:
            # 定義表格結構
            metadata = MetaData()
            tables = {
                'ohlc': Table('ohlc', metadata,
                    Column('date', DateTime),
                    Column('open', Float),
                    Column('high', Float),
                    Column('low', Float),
                    Column('close', Float),
                    Column('volume', Integer),
                    Column('n', Integer),
                    Column('vw', Float),
                    Column('timeframe', Text),
                    Column('RSI', Float),
                    Column('MACD', Float),
                    Column('MACD_signal', Float),
                    Column('MACD_hist', Float),
                    Column('ATR', Float),
                    Column('Stoch_k', Float),
                    Column('ADX', Float),
                    Column('Ichimoku_tenkan', Float),
                    Column('Ichimoku_kijun', Float),
                    Column('Ichimoku_span_a', Float),
                    Column('Ichimoku_span_b', Float),
                    Column('Ichimoku_cloud_top', Float),
                    Column('BB_upper', Float),
                    Column('BB_middle', Float),
                    Column('BB_lower', Float),
                    Column('EMA_12', Float),
                    Column('EMA_26', Float),
                    Column('fed_funds_rate', Float),
                    schema=None
                ),
                'economic': Table('economic_calendar', metadata,
                    Column('date', DateTime),
                    Column('event', Text),
                    Column('impact', Text),
                    Column('fed_funds_rate', Float),
                    schema=None
                ),
                'sentiment': Table('sentiment_data', metadata,
                    Column('date', DateTime),
                    Column('sentiment', Float),
                    schema=None
                ),
                'tweets': Table('tweets', metadata,
                    Column('date', DateTime),
                    Column('tweet_id', Text),
                    Column('text', Text),
                    Column('user_id', Text), 
                    schema=None
                ),
                'users': Table('users', metadata,
                    Column('user_id', Text),
                    Column('username', Text),
                    Column('verified', Integer),
                    schema=None
                ),
                'trades': Table('trades', metadata,
                    Column('id', Integer, primary_key=True, autoincrement=True),
                    Column('timestamp', DateTime),
                    Column('symbol', Text),
                    Column('price', Float),
                    Column('action', Text),
                    Column('volume', Float),
                    Column('stop_loss', Float),
                    Column('take_profit', Float),
                    schema=None
                )
            }

            # 驗證 data_type 是否有效
            if data_type not in tables:
                logging.error(f"無效的 data_type: {data_type}，必須是 {list(tables.keys())} 之一")
                return False

            # 先定義 df_to_save
            column_maps = {
                'ohlc': ['date', 'open', 'high', 'low', 'close', 'volume', 'n', 'vw', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'ATR', 'Stoch_k', 'ADX', 'Ichimoku_tenkan', 'Ichimoku_kijun', 'Ichimoku_span_a', 'Ichimoku_span_b', 'Ichimoku_cloud_top', 'BB_upper', 'BB_middle', 'BB_lower', 'EMA_12', 'EMA_26', 'fed_funds_rate'],
                'economic': ['date', 'event', 'impact', 'fed_funds_rate'],
                'sentiment': ['date', 'sentiment'],
                'tweets': ['date', 'tweet_id', 'text', 'user_id'],
                'users': ['user_id', 'username', 'verified'],
                'trades': ['id', 'timestamp', 'symbol', 'price', 'action', 'volume', 'stop_loss', 'take_profit']
            }
            columns = column_maps.get(data_type, [])
            if not columns:
                logging.error(f"無效的 column_map 對應 data_type: {data_type}")
                return False

            df_to_save = df[[col for col in columns if col in df.columns]].copy()
            if df_to_save.empty:
                logging.warning(f"無有效欄位可儲存，data_type={data_type}, 輸入數據欄位={df.columns.tolist()}")
                return False

            # 驗證輸入數據並填充缺失
            missing_cols = [col for col in columns if col not in df_to_save.columns]
            if missing_cols:
                logging.warning(f"數據缺少欄位 {missing_cols}，填充預設值 0.0")
                for col in missing_cols:
                    if col in tables[data_type].c:
                        default_val = tables[data_type].c[col].default.arg if tables[data_type].c[col].default else 0.0
                        df_to_save[col] = default_val

            if data_type == 'ohlc':
                df_to_save['timeframe'] = timeframe
            date_column = 'date' if data_type != 'trades' else 'timestamp'
            if date_column in df_to_save.columns:
                if not pd.api.types.is_datetime64_any_dtype(df_to_save[date_column]):
                    try:
                        df_to_save[date_column] = pd.to_datetime(df_to_save[date_column], errors='coerce')
                        invalid_dates = df_to_save[df_to_save[date_column].isna()]
                        if not invalid_dates.empty:
                            logging.warning(f"檢測到無效日期值: {invalid_dates[date_column].tolist()[:10]} (共 {len(invalid_dates)} 筆)")
                            df_to_save = df_to_save.dropna(subset=[date_column])
                    except Exception as e:
                        logging.error(f"日期轉換失敗: {str(e)}, traceback={traceback.format_exc()}")
                        return False
                else:
                    df_to_save[date_column] = df_to_save[date_column].apply(lambda x: x.to_pydatetime())
            
            if df_to_save.empty:
                logging.warning(f"無數據可儲存，data_type={data_type}, 輸入數據={df.head().to_dict()}")
                return False

            # 儲存數據
            engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", connect_args={"timeout": 30}, isolation_level="SERIALIZABLE")
            async with engine.begin() as conn:
                table = tables[data_type]
                for _, row in df_to_save.iterrows():
                    ins = insert(table).values(**row.to_dict()).prefix_with('OR REPLACE')
                    await conn.execute(ins)
                logging.info(f"增量儲存 {len(df_to_save)} 行 {data_type} 數據到 SQLite: timeframe={timeframe}")
            return True
        except Exception as e:
            if "database is locked" in str(e) and attempt < 2:
                logging.warning(f"資料庫鎖定，重試 {attempt + 1}/3")
                await asyncio.sleep(2 ** attempt * 2)
            else:
                logging.error(f"儲存 {data_type} 數據到 SQLite 失敗: {str(e)}, traceback={traceback.format_exc()}")
                return False
    return False

async def make_get_request(session: aiohttp.ClientSession, url: str, proxies: dict = None, **kwargs) -> aiohttp.ClientResponse:
    """處理 get 請求。"""
    if proxies is None:
        proxies = await get_proxy(load_settings())
    proxy_url = proxies.get('http') if proxies else None
    logging.info(f"發送 GET 請求: URL={url}, 代理={proxy_url or '無代理'}, 參數={kwargs.get('params', {})}")
    start_time = time.time()
    try:
        response = await session.get(url, proxy=proxy_url, **kwargs)
        logging.info(f"GET 請求成功: URL={url}, 狀態碼={response.status}, 耗時={time.time() - start_time:.2f}秒")
        return response
    except Exception as e:
        logging.error(f"GET 請求失敗: URL={url}, 錯誤={str(e)}, 耗時={time.time() - start_time:.2f}秒")
        raise

async def backup_database(db_path: str, root_dir: str):
    """備份資料庫。"""
    backup_dir = Path(root_dir) / 'backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / f"trading_data_{datetime.now().strftime('%Y%m%d')}.db"
    try:
        async with aiofiles.open(db_path, mode='rb') as src, aiofiles.open(backup_file, mode='wb') as dst:
            content = await src.read()
            if len(content) == 0:
                logging.error(f"備份失敗: 資料庫檔案 {db_path} 為空")
                return
            await dst.write(content)
        logging.info(f"資料庫已備份到 {backup_file}")
    except Exception as e:
        logging.error(f"資料庫備份失敗: {str(e)}, traceback={traceback.format_exc()}")

async def save_periodically(df_buffer: pd.DataFrame, timeframe: str, db_path: str, root_dir: str, data_type: str = 'ohlc'):
    """定期保存。"""
    save_interval = 1800 if timeframe == '1 hour' else 3 * 3600
    while True:
        try:
            if not df_buffer.empty:
                await save_data(df_buffer, timeframe, db_path, data_type)
                logging.info(f"定期儲存數據到 SQLite: timeframe={timeframe}, data_type={data_type}")
            if datetime.now().hour == 0 and datetime.now().minute < 5:
                await backup_database(db_path, root_dir)
                logging.info("資料庫已備份")
            await asyncio.sleep(save_interval)
        except Exception as e:
            logging.error(f"定期儲存失敗: {str(e)}, traceback={traceback.format_exc()}")

def load_settings():
    """載入設定。"""
    global _config_cache
    if _config_cache is not None:
        logging.info("從快取載入配置")
        return _config_cache
    config = {}
    root_dir = "C:\\Trading"
    config_dir = Path(root_dir) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    default_api_key = {}
    default_trading_params = {
        "max_position_size": 10000,
        "risk_per_trade": 0.01,
        "price_diff_threshold": {"high_volatility": 0.005, "normal": 0.003},
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "stoch_overbought": 80,
        "stoch_oversold": 20,
        "adx_threshold": 25,
        "obv_window": 14,
        "capital": 10000,
        "risk_percent": 0.01,
        "atr_threshold": 0.02,
        "min_backtest_days": 180,
        "ppo_learning_rate": 0.0003,
        "ppo_timesteps": 10000
    }
    default_system_config = {
        "data_source": "polygon",
        "symbol": "USDJPY=X",
        "timeframe": "1d",
        "root_dir": "C:\\Trading",
        "db_path": "C:\\Trading\\data\\trading_data.db",
        "proxies": {
            "http": "http://proxy1.scig.gov.hk:8080",
            "https": "http://proxy1.scig.gov.hk:8080"
        },
        "use_redis": False,
        "dependencies": [],
        "model_dir": "models",
        "model_periods": ["short_term", "medium_term", "long_term"],
        "offline_mode": False
    }
    config_files = {
        'api_key': (config_dir / 'api_key.json', default_api_key),
        'trading_params': (config_dir / 'trading_params.json', default_trading_params),
        'system_config': (config_dir / 'system_config.json', default_system_config)
    }
    try:
        for key, (file_path, default) in config_files.items():
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    config[key] = json.load(f)
                logging.info(f"成功載入配置文件: {file_path}")
            else:
                config[key] = default
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default, f, indent=4, ensure_ascii=False)
                logging.info(f"創建預設配置文件: {file_path}")
        if 'api_key' in config:
            config['api_key'].update({k: encrypt_key(v) for k, v in config['api_key'].items() if v})
        system_config = config.get('system_config', {})
        dependencies = system_config.get('dependencies', []) or ["pandas>=2.0.0",
                "yfinance>=0.2.0",
                "requests>=2.28.0",
                "textblob>=0.17.0",
                "torch>=2.0.0",
                "scikit-learn>=1.3.0",
                "xgboost>=2.0.0",
                "lightgbm>=4.0.0",
                "onnx>=1.14.0",
                "onnxruntime>=1.16.0",
                "transformers>=4.30.0",
                "stable-baselines3>=2.0.0",
                "pandas-ta>=0.3.0",
                "aiosqlite>=0.19.0",
                "gymnasium>=0.29.0",
                "python-dotenv>=1.0.0",
                "redis>=5.0.0",
                "streamlit>=1.25.0",
                "prometheus-client>=0.17.0",
                "ib-insync>=0.9.0",
                "cryptography>=41.0.0",
                "scipy>=1.10.0",
                "numpy>=1.24.0",
                "joblib>=1.3.0",
                "psutil>=5.9.0",
                "onnxmltools>=1.11.0",
                "onnxconverter-common>=1.13.0",
                "aiohttp>=3.8.0",
                "aiofiles>=23.1.0",
                "investpy>=1.0.0",
                "torch-directml>=0.2.0",
                "torch>=2.0.0",
                "pandas-ta>=0.3.14b0",
                "setuptools<81",
                "sqlalchemy>=2.0.43"
            ]
        system_config['dependencies'] = dependencies
        with open(config_files['system_config'][0], 'w', encoding='utf-8') as f:
            json.dump(system_config, f, indent=4, ensure_ascii=False)
        logging.info("已填充預設依賴到 system_config.json")
        requirements_path = Path(root_dir) / 'requirements.txt'
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(dependencies) + '\n')
        logging.info(f"已生成 requirements.txt: {requirements_path}")
        _config_cache = config
        logging.info("配置檔案載入成功")
        return config
    except Exception as e:
        logging.error(f"載入配置文件失敗: {str(e)}, traceback={traceback.format_exc()}")
        return {}

def check_hardware():
    """硬體檢測。"""
    try:
        import torch_directml
        device = torch_directml.device() if torch_directml.is_available() else torch.device('cpu')
        logging.info(f"使用裝置: {device}")
    except ImportError:
        device = torch.device('cpu')
        logging.info("回退到 CPU")
    providers = ort.get_available_providers()
    onnx_provider = next((p for p in ['VitisAIExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] if p in providers), 'CPUExecutionProvider')
    logging.info(f"ONNX provider: {onnx_provider}")
    device_config = {
        k: device if k in ['lstm', 'finbert', 'timeseries_transformer', 'distilbert', 'ppo'] else torch.device('cpu') for k in ['lstm', 'finbert', 'xgboost', 'randomforest', 'lightgbm', 'timeseries_transformer', 'distilbert', 'ppo']
    }
    lstm_path = 'models/lstm_model_quantized.onnx'
    session = None
    if os.path.exists(lstm_path):
        try:
            session = ort.InferenceSession(lstm_path, providers=[onnx_provider])
            logging.info(f"成功載入 ONNX 模型: {lstm_path}")
        except Exception as e:
            logging.warning(f"載入 ONNX 模型失敗: {lstm_path}, 錯誤={str(e)}")
    else:
        logging.warning(f"ONNX 模型檔案不存在: {lstm_path}")
    return device_config, session

async def test_proxy(proxy: dict) -> bool:
    """測試代理可用性，連續測試 3 次。"""
    logging.info("測試無代理模式" if not proxy else f"測試代理: {proxy}")
    test_url = "https://www.google.com"
    tasks = [test_single(proxy, test_url) for _ in range(3)]
    results = await asyncio.gather(*tasks)
    success_count = sum(results)
    success = success_count >= 2
    logging.info(f"代理測試{'成功' if success else '失敗'}: {proxy if proxy else '無代理'}，成功次數={success_count}/3")
    return success

async def test_single(proxy, test_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with (await make_get_request(session, test_url, proxies=proxy, timeout=5)) as response:
                if not response.ok:
                    logging.warning(f"代理測試響應非成功: 狀態碼={response.status}, URL={test_url}")
                return response.ok
    except Exception as e:
        logging.error(f"單次代理測試失敗: {str(e)}, URL={test_url}")
        return False

async def get_proxy(config: dict) -> dict:
    """獲取代理設置。"""
    global _proxy_cache, _proxy_tested
    if _proxy_cache is not None and _proxy_tested:
        logging.info("從快取載入代理設置")
        return _proxy_cache
    system_proxy = {k: os.getenv(k.upper() + '_PROXY') for k in ['http', 'https']}
    system_proxy = system_proxy if all(system_proxy.values()) else {}
    no_proxy = {}
    config_proxy = config.get('system_config', {}).get('proxies', {})
    proxy_options = [('配置文件代理', config_proxy), ('系統代理', system_proxy), ('無代理', no_proxy)]
    for name, proxy in proxy_options:
        if await test_proxy(proxy):
            _proxy_cache = proxy
            _proxy_tested = True
            logging.info(f"選擇代理: {name} -> {proxy}")
            if proxy:
                os.environ.update({k.upper() + '_PROXY': v for k, v in proxy.items()})
            else:
                [os.environ.pop(k, None) for k in ['HTTP_PROXY', 'HTTPS_PROXY']]
            return _proxy_cache
    _proxy_cache = {}
    _proxy_tested = False
    logging.error("所有代理測試失敗，無法建立網路連線")
    if config['system_config'].get('offline_mode', False):
        logging.info("進入離線模式，使用本地數據")
        return {}
    else:
        raise RuntimeError("無可用代理且未啟用離線模式")

def clear_proxy_cache():
    """清除代理快取。"""
    global _proxy_cache, _proxy_tested
    _proxy_cache = None
    _proxy_tested = None
    [os.environ.pop(k, None) for k in ['HTTP_PROXY', 'HTTPS_PROXY']]
    logging.info("代理快取及環境變數已清除")

def check_volatility(atr: float, low_threshold: float = 0.01, high_threshold: float = 0.02) -> str:
    """檢查波動。"""
    if atr > high_threshold:
        logging.warning("高波動偵測")
        return 'high'
    return 'medium' if atr > low_threshold else 'low'

def filter_future_dates(df: pd.DataFrame) -> pd.DataFrame:
    """過濾未來日期。"""
    if not df.empty and 'date' in df.columns:
        current_time = pd.to_datetime(datetime.now())
        initial_rows = len(df)
        df = df[df['date'] <= current_time].copy()
        logging.info(f"過濾未來日期，初始行數={initial_rows}，保留行數={len(df)}")
    return df

# 新增函數：檢查表是否有數據
async def check_table_data(db_path: str, table_name: str) -> dict:
    """檢查表是否有數據，返回數據量和最新日期。"""
    try:
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
        if table_name == 'trades':
            query = text(f"SELECT COUNT(*) as count, MAX(timestamp) as latest_date FROM {table_name}")
        else:
            query = text(f"SELECT COUNT(*) as count, MAX(date) as latest_date FROM {table_name}")
        async with engine.connect() as conn:
            result = await conn.execute(query)
            row = result.fetchone()
            count, latest_date = row[0], row[1]
            logging.info(f"{table_name} 表: 數據量={count}, 最新日期={latest_date or '無數據'}")
            return {"has_data": count > 0, "count": count, "latest_date": latest_date}
    except Exception as e:
        logging.error(f"檢查 {table_name} 表數據失敗: {str(e)}, traceback={traceback.format_exc()}")
        return {"has_data": False, "count": 0, "latest_date": None}
```

