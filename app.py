from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pickle, numpy as np, os, json, re, base64, io
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
from reportlab.lib.units import inch

app = Flask(__name__)
BASE = os.path.dirname(__file__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'cardioai-dev-key-2024')

# ── DATABASE ──────────────────────────────────────────────
# Use /data for Render persistent disk, fallback to local for dev
DB_PATH = os.environ.get('DB_PATH', os.path.join(BASE, 'cardioai.db'))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_PATH}" 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── RATE LIMITER ──────────────────────────────────────────
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# ── MODELS ────────────────────────────────────────────────
class Prediction(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)
    age         = db.Column(db.Float)
    sex         = db.Column(db.Float)
    probability = db.Column(db.Float)
    risk_level  = db.Column(db.String(20))
    prediction  = db.Column(db.Integer)
    input_data  = db.Column(db.Text)
    key_factors = db.Column(db.Text)
    shap_values = db.Column(db.Text)   # JSON - stored for report generation

class ChatMessage(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    timestamp     = db.Column(db.DateTime, default=datetime.utcnow)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction.id'), nullable=True)
    role          = db.Column(db.String(10))
    content       = db.Column(db.Text)

class Report(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction.id'))
    generated_at  = db.Column(db.DateTime, default=datetime.utcnow)
    filename      = db.Column(db.String(100))

# ── LOAD ML ARTIFACTS ─────────────────────────────────────
with open(os.path.join(BASE, 'models', 'cardioai_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE, 'models', 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE, 'models', 'feature_columns.pkl'), 'rb') as f:
    feature_columns = pickle.load(f)
with open(os.path.join(BASE, 'models', 'model_metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)
with open(os.path.join(BASE, 'models', 'shap_explainer.pkl'), 'rb') as f:
    explainer = pickle.load(f)

# ── GEMINI SETUP ──────────────────────────────────────────
GEMINI_KEY   = os.environ.get('GEMINI_API_KEY', '')
gemini_ready = False
gemini_client = None
if GEMINI_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_KEY)
        gemini_ready  = True
    except Exception:
        pass

# ── INPUT VALIDATION ──────────────────────────────────────
FIELD_RULES = {
    'age':      {'type': float, 'min': 1,   'max': 120, 'required': True},
    'sex':      {'type': float, 'min': 0,   'max': 1,   'required': True},
    'cp':       {'type': float, 'min': 1,   'max': 4,   'required': True},
    'trestbps': {'type': float, 'min': 50,  'max': 300, 'required': True},
    'chol':     {'type': float, 'min': 50,  'max': 700, 'required': True},
    'fbs':      {'type': float, 'min': 0,   'max': 1,   'required': True},
    'restecg':  {'type': float, 'min': 0,   'max': 2,   'required': True},
    'thalach':  {'type': float, 'min': 40,  'max': 250, 'required': True},
    'exang':    {'type': float, 'min': 0,   'max': 1,   'required': True},
    'oldpeak':  {'type': float, 'min': 0,   'max': 10,  'required': True},
    'slope':    {'type': float, 'min': 1,   'max': 3,   'required': True},
    'ca':       {'type': float, 'min': 0,   'max': 3,   'required': True},
    'thal':     {'type': float, 'min': 0,   'max': 7,   'required': True},
}

def validate_input(data):
    errors = []
    for field, rules in FIELD_RULES.items():
        if rules['required'] and (field not in data or data[field] == ''):
            errors.append(f"'{field}' is required.")
            continue
        try:
            val = rules['type'](data[field])
            if val < rules['min'] or val > rules['max']:
                errors.append(f"'{field}' must be between {rules['min']} and {rules['max']}.")
        except (ValueError, TypeError):
            errors.append(f"'{field}' must be a valid number.")
    return errors

# ── FEATURE VECTOR BUILDER ────────────────────────────────
def build_feature_vector(data):
    age=float(data['age']); sex=float(data['sex']); cp=float(data['cp'])
    trestbps=float(data['trestbps']); chol=float(data['chol']); fbs=float(data['fbs'])
    restecg=float(data['restecg']); thalach=float(data['thalach']); exang=float(data['exang'])
    oldpeak=float(data['oldpeak']); slope=float(data['slope']); ca=float(data['ca']); thal=float(data['thal'])
    row = {col: 0.0 for col in feature_columns}
    row.update({'age':age,'sex':sex,'trestbps':trestbps,'chol':chol,
                'fbs':fbs,'thalach':thalach,'exang':exang,'oldpeak':oldpeak,'ca':ca})
    for v in [0,1,2,3,4]:
        k=f'cp_{float(v)}'
        if k in row: row[k]=1.0 if cp==v else 0.0
    for v in [0,1,2]:
        k=f'restecg_{float(v)}'
        if k in row: row[k]=1.0 if restecg==v else 0.0
    for v in [0,1,2,3]:
        k=f'slope_{float(v)}'
        if k in row: row[k]=1.0 if slope==v else 0.0
    for v in [0,1,2,3,6,7]:
        k=f'thal_{float(v)}'
        if k in row: row[k]=1.0 if thal==v else 0.0
    return np.array([[row[col] for col in feature_columns]]), age, chol, trestbps, thalach, ca, exang

def get_key_factors(age, chol, trestbps, thalach, ca, exang):
    factors = []
    if age>55:       factors.append(f"Age {int(age)} — elevated risk zone")
    if chol>240:     factors.append(f"Cholesterol {int(chol)} mg/dl — above safe threshold")
    if trestbps>140: factors.append(f"Blood pressure {int(trestbps)} mmHg — hypertensive range")
    if thalach<120:  factors.append(f"Max heart rate {int(thalach)} bpm — low exercise capacity")
    if ca>0:         factors.append(f"{int(ca)} major vessel(s) with detected blockage")
    if exang==1:     factors.append("Exercise-induced angina present")
    if not factors:  factors.append("No major individual risk flags detected")
    return factors

# ── SHAP CHART GENERATOR ──────────────────────────────────
def generate_shap_chart(X_scaled, shap_vals):
    """Returns a base64 PNG string of SHAP waterfall chart."""
    try:
        # Pretty feature labels
        label_map = {
            'age':'Age','sex':'Sex','trestbps':'Resting BP','chol':'Cholesterol',
            'fbs':'Fasting BS','thalach':'Max HR','exang':'Exercise Angina',
            'oldpeak':'ST Depression','ca':'Major Vessels',
            'cp_0.0':'CP: Typical','cp_1.0':'CP: Atypical','cp_2.0':'CP: Non-anginal',
            'cp_3.0':'CP: Non-anginal2','cp_4.0':'CP: Asympt',
            'restecg_0.0':'ECG: Normal','restecg_1.0':'ECG: ST-T','restecg_2.0':'ECG: LVH',
            'slope_0.0':'Slope: 0','slope_1.0':'Slope: Up','slope_2.0':'Slope: Flat','slope_3.0':'Slope: Down',
            'thal_0.0':'Thal: 0','thal_1.0':'Thal: 1','thal_2.0':'Thal: 2',
            'thal_3.0':'Thal: Normal','thal_6.0':'Thal: Fixed','thal_7.0':'Thal: Reversable',
        }
        labels   = [label_map.get(c, c) for c in feature_columns]
        vals     = shap_vals[0]
        # Take top 10 by absolute value
        top_idx  = np.argsort(np.abs(vals))[-10:][::-1]
        top_vals = vals[top_idx]
        top_lbls = [labels[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        bar_colors = ['#ef4444' if v > 0 else '#22c55e' for v in top_vals]
        bars = ax.barh(range(len(top_vals)), top_vals[::-1], color=bar_colors[::-1], height=0.6)
        ax.set_yticks(range(len(top_vals)))
        ax.set_yticklabels(top_lbls[::-1], color='#94a3b8', fontsize=9)
        ax.set_xlabel('SHAP Value (impact on prediction)', color='#64748b', fontsize=8)
        ax.set_title('Feature Impact on Your Risk Score', color='#f1f5f9', fontsize=10, fontweight='bold', pad=10)
        ax.tick_params(colors='#64748b', labelsize=8)
        ax.spines['bottom'].set_color('#1e293b')
        ax.spines['left'].set_color('#1e293b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axvline(0, color='#334155', linewidth=0.8)
        ax.set_xlabel('← Reduces Risk  |  Increases Risk →', color='#64748b', fontsize=8)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                    facecolor='#111827', edgecolor='none')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"SHAP chart error: {e}")
        return None

# ── GEMINI FALLBACK ───────────────────────────────────────
def generate_fallback_response(msg):
    msg_lower = msg.lower()
    if any(w in msg_lower for w in ['cholesterol','chol']):
        return "Cholesterol above 240 mg/dl is considered high. Lifestyle changes like a low-saturated-fat diet, regular exercise, and sometimes medication can help manage it. Always consult your doctor for a personalised plan."
    if any(w in msg_lower for w in ['blood pressure','bp','hypertension']):
        return "Resting blood pressure above 140 mmHg is hypertensive. Reducing salt, exercising regularly, and managing stress are first-line interventions. Medication may also be needed — speak to your doctor."
    if any(w in msg_lower for w in ['exercise','workout','physical']):
        return "Regular aerobic exercise (150 min/week) significantly reduces cardiovascular risk. Brisk walking, cycling, or swimming are excellent."
    if any(w in msg_lower for w in ['diet','food','eat']):
        return "A heart-healthy diet includes vegetables, fruits, whole grains, lean proteins, and healthy fats. Limit processed foods, salt, and trans fats."
    if any(w in msg_lower for w in ['risk','score','probability','result']):
        return "Your CardioAI risk score estimates heart disease probability using 13 clinical features with a Random Forest model trained on real patient data."
    if any(w in msg_lower for w in ['hello','hi','hey']):
        return "Hello! I'm CardioAI's health assistant. Ask me about your results, risk factors, or general heart health."
    if any(w in msg_lower for w in ['shap','explain','feature','factor']):
        return "The SHAP chart shows which features pushed your risk score up (red bars) or down (green bars). Longer bars = bigger impact on your result."
    return "That's a great question about cardiovascular health. While I can provide general educational information, the best guidance comes from a qualified cardiologist. Is there something specific about your CardioAI assessment I can explain?"

# ── ROUTES ────────────────────────────────────────────────

@app.route('/')
def home():
    total = Prediction.query.count()
    high  = Prediction.query.filter_by(risk_level='High').count()
    low   = Prediction.query.filter_by(risk_level='Low').count()
    return render_template('index.html',
                           metrics=metadata['test_metrics'],
                           total_predictions=total,
                           high_risk=high,
                           low_risk=low)

@app.route('/predict', methods=['GET', 'POST'])
@limiter.limit("30 per minute")
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        # Validate
        errors = validate_input(data)
        if errors:
            return jsonify({'error': ' | '.join(errors)}), 422

        X, age, chol, trestbps, thalach, ca, exang = build_feature_vector(data)
        X_s   = scaler.transform(X)
        prob  = model.predict_proba(X_s)[0][1]
        pred  = int(model.predict(X_s)[0])
        risk  = 'Low' if prob < 0.3 else ('Moderate' if prob < 0.6 else 'High')
        color = '#22c55e' if risk == 'Low' else ('#f59e0b' if risk == 'Moderate' else '#ef4444')
        factors = get_key_factors(age, chol, trestbps, thalach, ca, exang)

        # SHAP — handle both list and 3D array output
        shap_vals = explainer.shap_values(X_s)
        if isinstance(shap_vals, list):
            sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif shap_vals.ndim == 3:
            sv = shap_vals[:, :, 1]   # class 1 (disease)
        else:
            sv = shap_vals
        shap_chart_b64 = generate_shap_chart(X_s, sv)
        shap_list = sv[0].tolist()

        # Top SHAP features for display
        label_map = {
            'age':'Age','sex':'Sex','trestbps':'Resting BP','chol':'Cholesterol',
            'fbs':'Fasting BS','thalach':'Max HR','exang':'Exercise Angina',
            'oldpeak':'ST Depression','ca':'Major Vessels',
        }
        top_shap = sorted(
            [{'feature': label_map.get(feature_columns[i], feature_columns[i]),
              'value': round(float(sv[0][i]), 4),
              'direction': 'risk' if sv[0][i] > 0 else 'protective'}
             for i in range(len(feature_columns))],
            key=lambda x: abs(x['value']), reverse=True
        )[:6]

        # Save to DB
        rec = Prediction(
            age=age, sex=float(data['sex']),
            probability=round(float(prob)*100, 1),
            risk_level=risk, prediction=pred,
            input_data=json.dumps(data),
            key_factors=json.dumps(factors),
            shap_values=json.dumps(shap_list)
        )
        db.session.add(rec); db.session.commit()

        return jsonify({
            'id':           rec.id,
            'probability':  round(float(prob)*100, 1),
            'prediction':   pred,
            'risk_level':   risk,
            'risk_color':   color,
            'key_factors':  factors,
            'roc_auc':      round(metadata['test_metrics']['ROC-AUC']*100, 1),
            'shap_chart':   shap_chart_b64,
            'top_shap':     top_shap,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/history/data')
@limiter.limit("60 per minute")
def history_data():
    records = Prediction.query.order_by(Prediction.timestamp.desc()).limit(100).all()
    return jsonify([{
        'id':          r.id,
        'timestamp':   r.timestamp.strftime('%d %b %Y, %H:%M'),
        'age':         int(r.age),
        'sex':         'Male' if r.sex == 1 else 'Female',
        'probability': r.probability,
        'risk_level':  r.risk_level,
        'prediction':  r.prediction,
        'key_factors': json.loads(r.key_factors) if r.key_factors else []
    } for r in records])

@app.route('/chat', methods=['GET'])
def chat_page():
    pred_id = request.args.get('pred_id')
    pred    = Prediction.query.get(pred_id) if pred_id else None
    return render_template('chat.html', pred=pred)

@app.route('/chat/message', methods=['POST'])
@limiter.limit("20 per minute")
def chat_message():
    try:
        data     = request.get_json()
        user_msg = data.get('message', '').strip()
        pred_id  = data.get('pred_id')
        history  = data.get('history', [])

        if not user_msg:
            return jsonify({'error': 'Empty message'}), 400
        if len(user_msg) > 1000:
            return jsonify({'error': 'Message too long (max 1000 chars)'}), 400

        context = ""
        if pred_id:
            pred = Prediction.query.get(pred_id)
            if pred:
                factors = json.loads(pred.key_factors) if pred.key_factors else []
                context = (
                    f"You are CardioAI's medical AI assistant. A patient received their heart disease risk assessment:\n"
                    f"- Risk Level: {pred.risk_level}\n"
                    f"- Disease Probability: {pred.probability}%\n"
                    f"- Age: {int(pred.age)}, Sex: {'Male' if pred.sex==1 else 'Female'}\n"
                    f"- Key Risk Factors: {', '.join(factors)}\n\n"
                    f"Be empathetic, educational, and always recommend consulting a real doctor for medical decisions. "
                    f"Answer in plain language for a general audience."
                )
        else:
            context = ("You are CardioAI's medical AI assistant specializing in cardiovascular health education. "
                       "Be empathetic, accurate, and always recommend consulting a qualified doctor.")

        if gemini_ready:
            messages = []
            for h in history[-6:]:
                role = 'user' if h['role'] == 'user' else 'model'
                messages.append(types.Content(role=role, parts=[types.Part(text=h['content'])]))
            full_prompt = f"{context}\n\nUser: {user_msg}" if not messages else user_msg
            messages.append(types.Content(role='user', parts=[types.Part(text=full_prompt)]))
            response = gemini_client.models.generate_content(
                model='gemini-1.5-flash',
                contents=messages
            )
            reply = response.text
        else:
            reply = generate_fallback_response(user_msg)

        if pred_id:
            db.session.add(ChatMessage(prediction_id=pred_id, role='user', content=user_msg))
            db.session.add(ChatMessage(prediction_id=pred_id, role='ai',   content=reply))
            db.session.commit()

        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/report/<int:pred_id>')
@limiter.limit("10 per minute")
def download_report(pred_id):
    pred        = Prediction.query.get_or_404(pred_id)
    input_data  = json.loads(pred.input_data)  if pred.input_data  else {}
    key_factors = json.loads(pred.key_factors) if pred.key_factors else []
    shap_list   = json.loads(pred.shap_values) if pred.shap_values else None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=0.7*inch, rightMargin=0.7*inch,
                            topMargin=0.7*inch,  bottomMargin=0.7*inch)
    story = []

    # Styles
    title_s = ParagraphStyle('T', fontSize=22, fontName='Helvetica-Bold',
                              textColor=colors.HexColor('#e8394a'), spaceAfter=4)
    sub_s   = ParagraphStyle('S', fontSize=9,  fontName='Helvetica',
                              textColor=colors.HexColor('#64748b'), spaceAfter=14)
    h2_s    = ParagraphStyle('H', fontSize=12, fontName='Helvetica-Bold',
                              textColor=colors.HexColor('#1e293b'), spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle('B', fontSize=9,  fontName='Helvetica',
                              textColor=colors.HexColor('#334155'), leading=14)
    disc_s  = ParagraphStyle('D', fontSize=7.5,fontName='Helvetica',
                              textColor=colors.HexColor('#94a3b8'), leading=11)

    story.append(Paragraph("CardioAI — Heart Disease Risk Report", title_s))
    story.append(Paragraph(
        f"Generated: {pred.timestamp.strftime('%d %B %Y at %H:%M UTC')}  ·  Assessment ID: #{pred.id}  ·  Model: Random Forest (ROC-AUC 0.86)",
        sub_s))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#e8394a'), spaceAfter=14))

    # Risk Summary Table
    rc  = '#22c55e' if pred.risk_level=='Low' else ('#f59e0b' if pred.risk_level=='Moderate' else '#ef4444')
    t   = Table([
            ['Risk Level', 'Probability', 'Outcome', 'Prediction Model'],
            [pred.risk_level, f"{pred.probability}%",
             'Disease Likely' if pred.prediction==1 else 'No Disease Detected', 'Random Forest']
          ], colWidths=[1.55*inch]*4)
    t.setStyle(TableStyle([
        ('BACKGROUND',  (0,0),(-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR',   (0,0),(-1,0), colors.white),
        ('FONTNAME',    (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0),(-1,0), 8),
        ('FONTSIZE',    (0,1),(-1,1), 11),
        ('FONTNAME',    (0,1),(0,1),  'Helvetica-Bold'),
        ('TEXTCOLOR',   (0,1),(0,1),  colors.HexColor(rc)),
        ('ALIGN',       (0,0),(-1,-1),'CENTER'),
        ('VALIGN',      (0,0),(-1,-1),'MIDDLE'),
        ('BACKGROUND',  (0,1),(-1,1), colors.HexColor('#f8fafc')),
        ('GRID',        (0,0),(-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWHEIGHT',   (0,0),(-1,-1), 26),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Clinical Values
    story.append(Paragraph("Clinical Input Values", h2_s))
    cp_m = {'1':'Typical Angina','2':'Atypical Angina','3':'Non-anginal Pain','4':'Asymptomatic'}
    thal_m = {'3':'Normal','6':'Fixed Defect','7':'Reversable Defect'}
    ecg_m = {'0':'Normal','1':'ST-T Abnormality','2':'LV Hypertrophy'}
    slp_m = {'1':'Upsloping','2':'Flat','3':'Downsloping'}
    rows = [
        ['Parameter','Value','Parameter','Value'],
        ['Age', f"{input_data.get('age','—')} yrs", 'Cholesterol', f"{input_data.get('chol','—')} mg/dl"],
        ['Sex', 'Male' if str(input_data.get('sex'))=='1' else 'Female', 'Fasting Blood Sugar', 'High' if str(input_data.get('fbs'))=='1' else 'Normal'],
        ['Resting BP', f"{input_data.get('trestbps','—')} mmHg", 'Max Heart Rate', f"{input_data.get('thalach','—')} bpm"],
        ['Chest Pain', cp_m.get(str(input_data.get('cp','')),'—'), 'Exercise Angina', 'Yes' if str(input_data.get('exang'))=='1' else 'No'],
        ['Resting ECG', ecg_m.get(str(input_data.get('restecg','')),'—'), 'ST Depression', f"{input_data.get('oldpeak','—')}"],
        ['ST Slope', slp_m.get(str(input_data.get('slope','')),'—'), 'Major Vessels', f"{input_data.get('ca','—')}"],
        ['Thalassemia', thal_m.get(str(input_data.get('thal','')),'—'), '', ''],
    ]
    ct = Table(rows, colWidths=[1.65*inch, 1.4*inch, 1.65*inch, 1.4*inch])
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0),  colors.HexColor('#1e293b')),
        ('TEXTCOLOR',  (0,0),(-1,0),  colors.white),
        ('FONTNAME',   (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 8.5),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#f8fafc')]),
        ('FONTNAME',   (0,1),(0,-1),  'Helvetica-Bold'),
        ('FONTNAME',   (2,1),(2,-1),  'Helvetica-Bold'),
        ('TEXTCOLOR',  (0,1),(0,-1),  colors.HexColor('#475569')),
        ('TEXTCOLOR',  (2,1),(2,-1),  colors.HexColor('#475569')),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWHEIGHT',  (0,0),(-1,-1), 20),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1), 8),
    ]))
    story.append(ct)
    story.append(Spacer(1, 12))

    # Key factors
    story.append(Paragraph("Key Contributing Risk Factors", h2_s))
    for factor in key_factors:
        story.append(Paragraph(f"• {factor}", body_s))
    story.append(Spacer(1, 12))

    # SHAP chart in PDF
    if shap_list:
        try:
            sv_arr = np.array(shap_list).reshape(1, -1)
            chart_b64 = generate_shap_chart(None, sv_arr)
            if chart_b64:
                story.append(Paragraph("SHAP Explainability — Feature Impact", h2_s))
                story.append(Paragraph(
                    "Red bars indicate features that increased the disease risk probability. "
                    "Green bars indicate protective features that reduced the risk score.", body_s))
                story.append(Spacer(1, 6))
                img_data = base64.b64decode(chart_b64)
                img_buf  = io.BytesIO(img_data)
                story.append(RLImage(img_buf, width=5.5*inch, height=3.2*inch))
                story.append(Spacer(1, 12))
        except Exception:
            pass

    # Recommendations
    story.append(Paragraph("General Recommendations", h2_s))
    recs = []
    if pred.risk_level == 'High':
        recs = [
            "Consult a cardiologist as soon as possible for a comprehensive evaluation.",
            "Monitor blood pressure and cholesterol levels regularly.",
            "Adopt a heart-healthy diet: reduce saturated fats, salt, and processed foods.",
            "Begin a supervised exercise program if cleared by your doctor.",
            "Avoid smoking and limit alcohol consumption."
        ]
    elif pred.risk_level == 'Moderate':
        recs = [
            "Schedule a check-up with your doctor to discuss your cardiovascular risk.",
            "Aim for 150 minutes of moderate aerobic exercise per week.",
            "Reduce dietary cholesterol and increase fibre intake.",
            "Monitor blood pressure and maintain a healthy weight.",
            "Manage stress through mindfulness, yoga, or relaxation techniques."
        ]
    else:
        recs = [
            "Maintain your current healthy lifestyle and continue regular health check-ups.",
            "Keep active with at least 150 minutes of moderate exercise per week.",
            "Follow a balanced, heart-healthy diet rich in vegetables and whole grains.",
            "Avoid smoking and limit alcohol.",
            "Get annual check-ups to monitor blood pressure and cholesterol."
        ]
    for r in recs:
        story.append(Paragraph(f"• {r}", body_s))
    story.append(Spacer(1, 14))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceAfter=8))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This report is generated by an AI/ML model for educational and research purposes only. "
        "It does not constitute medical advice and must not replace consultation with a qualified healthcare professional. "
        "CardioAI uses a Random Forest classifier (ROC-AUC: 0.86) trained on the UCI Heart Disease dataset.",
        disc_s))

    doc.build(story)
    buf.seek(0)

    # Log report in DB
    fn = f"CardioAI_Report_{pred_id}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
    db.session.add(Report(prediction_id=pred_id, filename=fn))
    db.session.commit()

    return send_file(buf, mimetype='application/pdf', as_attachment=True, download_name=fn)

# ── ERROR HANDLERS ────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({'error': 'Too many requests. Please wait a moment and try again.'}), 429

# ── INIT ──────────────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    port  = int(os.environ.get('PORT', 5000))
    app.run(debug=debug, host='0.0.0.0', port=port)
