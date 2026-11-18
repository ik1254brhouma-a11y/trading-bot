import streamlit as st
import ccxt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="لوحة تحكم روبوت التداول الذكي", page_icon="🤖", layout="centered")

st.title("🤖 لوحة تحكم روبوت التداول بالذكاء الاصطناعي")
st.write("تحكم في إعدادات الروبوت، راقب السوق، ونفّذ الصفقات بضغطة زر.")

st.sidebar.header("⚙️ إعدادات الربط والتداول")
api_key = st.sidebar.text_input("مفتاح API Key", type="password")
secret_key = st.sidebar.text_input("مفتاح Secret Key", type="password")
use_testnet = st.sidebar.checkbox("استخدام الوضع التجريبي الآمن (Testnet)", value=True)

symbol = st.sidebar.selectbox("زوج العملات", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
timeframe = st.sidebar.selectbox("الإطار الزمني", ["5m", "15m", "1h", "4h"], index=1)
trade_amount = st.sidebar.number_input("كمية الشراء لكل صفقة", min_value=0.0001, value=0.001, format="%.4f")

def analyze_market(exchange, symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'Price_Change']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[features], df['Target'])
    
    latest_row = df.iloc[-1][features].values.reshape(1, -1)
    prediction = model.predict(latest_row)[0]
    current_price = df.iloc[-1]['Close']
    
    return prediction, current_price, df

if st.button("🚀 فحص السوق وتشغيل الروبوت الآن"):
    if not api_key or not secret_key:
        st.error("⚠️ يرجى إدخال مفاتيح API Key و Secret Key في القائمة الجانبية أولاً.")
    else:
        try:
            with st.spinner("جاري الاتصال بالمنصة وتحليل البيانات بالذكاء الاصطناعي..."):
                exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': secret_key,
                    'enableRateLimit': True,
                })
                if use_testnet:
                    exchange.set_sandbox_mode(True)
                
                balance = exchange.fetch_balance()
                usdt_balance = balance['USDT']['free']
                
                prediction, current_price, df = analyze_market(exchange, symbol, timeframe)
                
                col1, col2 = st.columns(2)
                col1.metric("💰 رصيدك المتاح (USDT)", f"{usdt_balance:.2f} $")
                col2.metric(f"📈 سعر {symbol} الحالي", f"{current_price:.2f} $")
                
                st.subheader("حركة السعر الأخيرة")
                st.line_chart(df['Close'])
                
                st.subheader("🧠 قرار الذكاء الاصطناعي")
                if prediction == 1:
                    st.success("إشارة: [ شراء - BUY ] 🟢 — السوق يظهر إيجابية.")
                else:
                    st.warning("إشارة: [ انتظار - HOLD ] 🔴 — لا توجد فرصة شراء قوية حالياً.")
                    
        except Exception as e:
            st.error(f"حدث خطأ أثناء الاتصال: {e}")

