import csv
import os
import re
import sqlite3
import uuid
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask, Response, flash, g, jsonify, redirect, render_template,
    request, session, stream_with_context, url_for,
)
from openai import OpenAI
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
# override=True: 디버그 리로더의 부모 프로세스가 물려준 (빈) 값보다 .env 를 우선한다
load_dotenv(ENV_PATH, override=True)

DB_PATH = BASE_DIR / "users.db"            # 회원정보 (SQLite)
PRODUCTS_CSV = BASE_DIR / "static" / "예금목록.csv"   # 금융상품 (예금 목록)
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "gif", "webp"}

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY") or "dev-secret-change-me"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 프로필 사진 최대 5MB

_client_cache = {"key": None, "client": None}


def get_openai():
    """.env 를 매번 다시 읽어서, 키를 바꿔도 서버 재시작 없이 반영한다."""
    load_dotenv(ENV_PATH, override=True)
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        return None
    if _client_cache["key"] != key:
        _client_cache.update(key=key, client=OpenAI(api_key=key))
    return _client_cache["client"]


def openai_model():
    return os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

# ---------------------------------------------------------------- 선택지
GENDERS = ["남성", "여성"]
RISK_LEVELS = ["안정형", "안정추구형", "위험중립형", "적극투자형", "공격투자형"]
JOBS = ["직장인", "공무원", "전문직", "자영업", "프리랜서", "학생", "주부", "무직/은퇴"]
INCOME_OPTIONS = [  # (저장값: 연소득 만원, 표시 문구)
    (1500, "2천만원 미만"),
    (3000, "2천만원 ~ 4천만원"),
    (5000, "4천만원 ~ 6천만원"),
    (7000, "6천만원 ~ 8천만원"),
    (9000, "8천만원 ~ 1억원"),
    (12000, "1억원 이상"),
]
INCOME_LABELS = dict(INCOME_OPTIONS)


# ---------------------------------------------------------------- DB
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name          TEXT NOT NULL,
                email         TEXT NOT NULL,
                age           INTEGER NOT NULL,
                gender        TEXT NOT NULL,
                risk          TEXT NOT NULL,
                job           TEXT NOT NULL,
                income        INTEGER NOT NULL,
                profile_image TEXT,
                created_at    TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


def current_user():
    uid = session.get("user_id")
    if uid is None:
        return None
    user = get_db().execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    if user is None:
        session.clear()
    return user


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if user is None:
            if request.path.startswith("/api/"):
                return jsonify(error="로그인이 필요합니다."), 401
            return redirect(url_for("login"))
        g.user = user
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_helpers():
    return {"income_label": lambda v: INCOME_LABELS.get(v, f"{v}만원")}


# ---------------------------------------------------------------- 상품 / 추천
# 예금목록.csv 는 네이버페이 예적금 페이지를 수집한 데이터라 연령·직업·성향 컬럼이 없다.
# 상품명·가입대상·가입방법·상세정보(우대조건)에서 특징 태그를 뽑아 추천에 쓴다.
_RE_AGE_MIN = re.compile(r"만\s*(\d+)\s*세\s*이상")
_RE_AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(백만|천만|억|만|백|천)?\s*원")
_AMOUNT_UNIT = {"": 1 / 10000, "천": 0.1, "백": 0.01, "만": 1, "백만": 100, "천만": 1000, "억": 10000}
_RE_UPDATED = re.compile(r"최종 업데이트 (\d{4}\.\d{2}\.\d{2})")
_RE_RATE_TYPE = re.compile(r"\| 유형 ([^|]+?)\s*(?:\||$)")
_RE_COND_RATE = re.compile(r"\s*[:：]\s*(?:연\s*)?(\d+(?:\.\d+)?)\s*%p?\s*$")
_BADGES = ("특판", "방문없이가입", "누구나가입")
_GENERIC_NAMES = {"정기예금", "일반정기예금", "실세금리정기예금", "회전정기예금"}

TAG_LABELS = {
    "pension": "연금 손님 전용", "pension_bonus": "연금 수령 우대", "first": "첫 거래 우대",
    "salary": "주거래·급여 우대", "business": "개인사업자 가입 가능", "monthly": "매월 이자 지급",
    "prepaid": "이자 먼저 지급", "variable": "회전·변동금리", "flexible": "자유만기·중도해지 유리",
    "special": "특판", "premium": "고액 예치", "small": "소액 가입", "digital_only": "비대면 전용",
    "simple": "우대조건 없음",
}

_products_cache = {"mtime": None, "items": []}


def _short(text, n):
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _min_amount(text):
    """가입금액 문구에서 가장 작은 금액(만원)을 찾는다."""
    amounts = [
        float(num.replace(",", "")) * _AMOUNT_UNIT[unit or ""]
        for num, unit in _RE_AMOUNT.findall(text or "")
    ]
    return min(amounts) if amounts else None


def _core_detail(text):
    """상세정보전체에서 네이버 메뉴·광고·계산기 부분을 걷어낸다."""
    start = text.find("전체메뉴 | ")
    s = text[start + len("전체메뉴 | "):] if start >= 0 else text
    end = s.find(" | 금리 높은 예금")
    if end >= 0:
        s = s[:end]
    s = re.sub(r"12개월 만기시 세후수령액.*?세후수령액 \| [\d,]+원 \| ", "", s)
    for junk in ("전화 문의 | ", "공식홈에서 더 알아보기 | ", "더보기 | "):
        s = s.replace(junk, "")
    return s.strip()


def _conditions(core):
    m = re.search(r"조건별 \| (.*?)(?: \| 유형 |$)", core)
    if not m:
        return []
    items = []
    for part in m.group(1).split(" | "):
        part = part.strip(" -ㆍ·*※")
        if part and not part.isdigit() and part != "금리우대쿠폰":
            items.append(part)
    return items


def _benefit_lines(conditions, k=3):
    """카드에 보여줄 우대조건: '조건 : 0.2%' 형태를 우선 사용한다."""
    rated = [(c, _RE_COND_RATE.search(c)) for c in conditions]
    rated = [(c[: m.start()], m.group(1)) for c, m in rated if m]
    if rated:
        return [f"{_short(label, 34)} +{rate}%p" for label, rate in rated[:k]]
    return [_short(c, 40) for c in conditions[:k]]


def _tags(p, conditions):
    name, target, method = p["name"], p["target"], p["method"]
    cond = " ".join(conditions)
    t = set()
    if re.search(r"연금|골든|시니어|에이지|knowhow", name, re.I):
        t.add("pension")
    elif "연금" in cond and re.search(r"수령|수급|받은|입금", cond):
        t.add("pension_bonus")
    if re.search(r"첫거래|첫 거래|첫만남|굿스타트|첫 로그인|보유하지 않은", name + target + cond):
        t.add("first")
    if re.search(r"주거래|급여", name + cond):
        t.add("salary")
    if ("개인사업자" in target and not re.search(r"개인사업자\s*제외", target)) or re.search(r"사업자|창업|가맹점", name):
        t.add("business")
    if re.search(r"월이자|매월|달달|월 이자", name):
        t.add("monthly")
    if re.search(r"이자먼저|먼저 이자|선이자", name + p["interest"]):
        t.add("prepaid")
    if "변동" in p["rate_type"] or re.search(r"회전|연동|실세|시장금리|CD", name):
        t.add("variable")
    if re.search(r"자유만기|중도해지|내맘대로|자유적립", name):
        t.add("flexible")
    if "특판" in p["badges"] or "특판" in name:
        t.add("special")
    if re.search(r"고단위|프리미엄", name) or (p["min_amount"] or 0) >= 500:
        t.add("premium")
    if p["min_amount"] is not None and p["min_amount"] <= 10:
        t.add("small")
    online = "방문없이가입" in p["badges"] or re.search(r"인터넷|스마트|모바일|앱|비대면|뱅킹|온라인|Web", method)
    if (online and not re.search(r"영업점|지점|창구", method)) or re.search(r"카카오뱅크|케이뱅크|토스뱅크", p["bank"]):
        t.add("digital_only")
    if p["max_rate"] - p["base_rate"] <= 0.1:
        t.add("simple")
    return t


def load_products():
    """예금목록.csv 를 읽는다. 파일이 바뀌면 재시작 없이 다시 읽는다."""
    mtime = PRODUCTS_CSV.stat().st_mtime
    if _products_cache["mtime"] == mtime:
        return _products_cache["items"]

    items = []
    with open(PRODUCTS_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            detail = row.get("상세정보전체", "")
            core = _core_detail(detail)
            # 머리말: '상품명 | (네이버 페이 |) 금융사 | 특판방문없이가입누구나가입 | 이율 ...'
            head = " ".join(core.split(" | 이율 |", 1)[0].split(" | ")[1:])
            rate_type = _RE_RATE_TYPE.search(core)
            updated = _RE_UPDATED.search(detail)
            age_min = _RE_AGE_MIN.search(row["가입대상"])
            url = row.get("상세URL", "")
            p = {
                "id": url.rstrip("/").rsplit("/", 1)[-1] or row["상품명"],
                "bank": row["금융사"].strip(),
                "name": row["상품명"].strip(),
                "category": "정기예금",
                "max_rate": float(row["최고금리_숫자"] or row["최고금리"] or 0),
                "base_rate": float(row["기본금리_숫자"] or row["기본금리"] or 0),
                "term": row["가입기간"].strip() or "-",
                "amount": row["가입금액"].strip() or "-",
                "method": row["가입방법"].strip() or "-",
                "target": row["가입대상"].strip() or "-",
                "interest": row.get("이자지급", "").strip(),
                "rate_type": rate_type.group(1).strip() if rate_type else "",
                "badges": [b for b in _BADGES if b in head],
                "url": url,
                "core": core,
                "updated": updated.group(1) if updated else "",
                "age_min": int(age_min.group(1)) if age_min else 0,
                "min_amount": _min_amount(row["가입금액"]),
            }
            conditions = _conditions(core)
            p["benefits"] = _benefit_lines(conditions)
            p["tags"] = _tags(p, conditions)
            gap = p["max_rate"] - p["base_rate"]
            p["description"] = (
                f"우대조건 없이 누구나 연 {p['base_rate']:.2f}%를 받는 예금"
                if "simple" in p["tags"]
                else f"우대조건을 채우면 최대 +{gap:.2f}%p 금리가 더해지는 예금"
            )
            items.append(p)

    _products_cache.update(mtime=mtime, items=items)
    return items


def products_updated(products):
    return max((p["updated"] for p in products), default="")


# 투자성향별 (기본금리 가중치, 최고금리 가중치): 안정형일수록 조건 없이 받는 기본금리를 중시
RISK_WEIGHTS = {
    "안정형": (2.0, 0.5), "안정추구형": (1.5, 1.0), "위험중립형": (1.0, 1.5),
    "적극투자형": (0.5, 2.0), "공격투자형": (0.3, 2.2),
}


def recommend(user, products, n=3):
    """가입 가능 연령으로 거른 뒤 투자성향·나이·직업·소득 점수로 상위 n개를 고른다 (은행당 1개)."""
    age, risk, job, income = user["age"], user["risk"], user["job"], user["income"]
    wb, wm = RISK_WEIGHTS.get(risk, (1.0, 1.0))
    scored = []
    for p in products:
        if age < p["age_min"]:
            continue
        t, amt = p["tags"], p["min_amount"]
        bonus = []  # (점수, 추천 이유)

        def add(cond, pts, reason=None):
            if cond:
                bonus.append((pts, reason))

        # 투자성향
        if risk in ("안정형", "안정추구형"):
            add("고정" in p["rate_type"] and "variable" not in t, 1.0, "고정금리로 안정적")
            add("simple" in t, 1.0 if risk == "안정형" else 0.5, "조건 없이 기본금리 그대로")
            add("variable" in t, -1.0)
        elif risk == "위험중립형":
            add("variable" in t, 1.5, "금리 변동에 유연한 회전·연동형")
        else:
            add("flexible" in t, 2.0, "자유로운 만기·중도해지")
            add("special" in t, 1.0, "특판 고금리")
            add("simple" in t, -0.5)

        # 나이
        if age < 30:
            add("first" in t, 1.5, "첫 거래 우대")
            add("digital_only" in t, 1.0, "비대면 간편 가입")
        if age >= 50:
            add("pension" in t, 2.5, "연금 손님 전용")
            add("pension_bonus" in t, 1.0, "연금 수령 우대")
            add("monthly" in t, 1.5, "매월 이자 지급")
        if age < 50:
            add("pension" in t, -3.0)

        # 직업
        if job in ("직장인", "공무원", "전문직"):
            add("salary" in t, 1.5, "급여이체 우대")
        elif job in ("자영업", "프리랜서"):
            add("business" in t, 2.0 if job == "자영업" else 1.0, "개인사업자 가입 가능")
        elif job == "무직/은퇴":
            add("pension" in t or "pension_bonus" in t, 1.5, "연금 수령 우대")
            add("monthly" in t, 1.5, "매월 이자 지급")
            add("prepaid" in t, 1.0, "이자 먼저 지급")
        elif job == "학생":
            add("small" in t, 2.0, "소액으로 가입 가능")
            add("digital_only" in t, 1.0, "비대면 간편 가입")
            add(amt is not None and amt > 100, -2.0)
        elif job == "주부":
            add("monthly" in t, 1.0, "매월 이자 지급")
            add("simple" in t, 0.5, "조건 없이 기본금리 그대로")

        # 소득
        if income <= 1500:
            add("small" in t, 1.5, "소액으로 가입 가능")
            add(amt is not None and amt >= 300, -1.5)
        if income >= 9000:
            add("premium" in t, 1.5, "고액 예치에 적합")

        score = wb * p["base_rate"] + wm * p["max_rate"] + sum(pts for pts, _ in bonus)
        reasons = []
        for pts, reason in sorted(bonus, key=lambda b: -b[0]):
            if pts > 0 and reason and reason not in reasons:
                reasons.append(reason)
        if p["max_rate"] >= 3.8:
            reasons.append("최고 수준 금리")
        scored.append((score, {**p, "reasons": reasons[:3]}))

    scored.sort(key=lambda x: x[0], reverse=True)
    picks, banks = [], set()
    for _, p in scored:
        if p["bank"] in banks:
            continue
        picks.append(p)
        banks.add(p["bank"])
        if len(picks) == n:
            break
    return picks


# ---------------------------------------------------------------- 인증
@app.route("/")
def index():
    return redirect(url_for("main" if current_user() else "login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = get_db().execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            return redirect(url_for("main"))
        flash("아이디 또는 비밀번호가 올바르지 않습니다.", "error")
        return render_template("login.html", username=username)
    if current_user():
        return redirect(url_for("main"))
    return render_template("login.html", username="")


@app.route("/logout")
def logout():
    session.clear()
    flash("안전하게 로그아웃되었습니다.", "info")
    return redirect(url_for("login"))


def _validate_signup(form):
    errors = []
    if not re.fullmatch(r"[A-Za-z0-9_]{4,20}", form.get("username", "")):
        errors.append("아이디는 영문·숫자·_ 조합 4~20자로 입력해 주세요.")
    if len(form.get("password", "")) < 8:
        errors.append("비밀번호는 8자 이상이어야 합니다.")
    if form.get("password") != form.get("password_confirm"):
        errors.append("비밀번호 확인이 일치하지 않습니다.")
    if not form.get("name", "").strip():
        errors.append("이름을 입력해 주세요.")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", form.get("email", "")):
        errors.append("올바른 이메일 주소를 입력해 주세요.")
    try:
        if not 14 <= int(form.get("age", "")) <= 120:
            raise ValueError
    except ValueError:
        errors.append("나이는 14~120 사이의 숫자로 입력해 주세요.")
    if form.get("gender") not in GENDERS:
        errors.append("성별을 선택해 주세요.")
    if form.get("risk") not in RISK_LEVELS:
        errors.append("투자성향을 선택해 주세요.")
    if form.get("job") not in JOBS:
        errors.append("직업을 선택해 주세요.")
    if form.get("income", "") not in {str(v) for v, _ in INCOME_OPTIONS}:
        errors.append("연소득 구간을 선택해 주세요.")
    return errors


def _save_profile_image(file):
    """저장된 파일명을 돌려준다. 파일이 없으면 None, 형식이 틀리면 ValueError."""
    if not file or not file.filename:
        return None
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT:
        raise ValueError("프로필 사진은 png, jpg, gif, webp 형식만 가능합니다.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.{ext}"
    file.save(UPLOAD_DIR / filename)
    return filename


@app.route("/signup", methods=["GET", "POST"])
def signup():
    options = dict(genders=GENDERS, risks=RISK_LEVELS, jobs=JOBS, incomes=INCOME_OPTIONS)
    if request.method == "GET":
        return render_template("signup.html", form={}, **options)

    form = request.form
    errors = _validate_signup(form)
    db = get_db()
    if not errors and db.execute(
        "SELECT 1 FROM users WHERE username = ?", (form["username"],)
    ).fetchone():
        errors.append("이미 사용 중인 아이디입니다.")

    image = None
    if not errors:
        try:
            image = _save_profile_image(request.files.get("profile_image"))
        except ValueError as e:
            errors.append(str(e))

    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("signup.html", form=form, **options)

    db.execute(
        """INSERT INTO users (username, password_hash, name, email, age, gender,
                              risk, job, income, profile_image)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            form["username"], generate_password_hash(form["password"]),
            form["name"].strip(), form["email"].strip(), int(form["age"]),
            form["gender"], form["risk"], form["job"], int(form["income"]), image,
        ),
    )
    db.commit()
    flash("회원가입이 완료되었습니다. 로그인해 주세요.", "info")
    return redirect(url_for("login"))


@app.errorhandler(413)
def too_large(_e):
    flash("프로필 사진은 5MB 이하만 업로드할 수 있습니다.", "error")
    return redirect(url_for("signup"))


# ---------------------------------------------------------------- 메인 / 챗봇
@app.route("/main")
@login_required
def main():
    products = load_products()
    picks = recommend(g.user, products)
    return render_template(
        "main.html", user=g.user, products=picks,
        total=len(products), updated=products_updated(products),
    )


@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html", user=g.user, ai_ready=get_openai() is not None)


def _mentioned(products, text, limit=4):
    """질문에 상품명이 들어 있으면 그 상품의 상세정보를 프롬프트에 넣는다."""
    q = re.sub(r"\s+", "", text)
    found = []
    for p in products:
        name = re.sub(r"\s+", "", p["name"])
        bank_hit = re.sub(r"\s+", "", p["bank"]) in q
        if name in q and (bank_hit or (len(name) >= 5 and name not in _GENERIC_NAMES)):
            found.append(p)
    found.sort(key=lambda p: -len(p["name"]))  # 더 구체적인 이름 우선
    return found[:limit]


def _system_prompt(user, products, question):
    picks = recommend(user, products)
    pick_ids = {p["id"] for p in picks}
    lines = []
    for p in sorted(products, key=lambda p: -p["max_rate"]):
        mark = " ★손님 추천 TOP3" if p["id"] in pick_ids else ""
        features = ", ".join(TAG_LABELS[t] for t in sorted(p["tags"]) if t in TAG_LABELS)
        lines.append(
            f"- {p['bank']} 「{p['name']}」{mark} | 최고 연 {p['max_rate']:.2f}% / 기본 연 {p['base_rate']:.2f}% | "
            f"기간 {_short(p['term'], 40)} | 금액 {_short(p['amount'], 30)} | 가입 {_short(p['method'], 30)} | "
            f"대상 {_short(p['target'], 35)} | {p['rate_type'] or '-'} | 특징 {features or '-'}"
        )

    detail_items = {p["id"]: p for p in [*_mentioned(products, question), *picks]}.values()
    details = "\n\n".join(
        f"### {p['bank']} 「{p['name']}」\n{p['core']}\n상세 페이지: {p['url']}" for p in detail_items
    )

    return f"""당신은 FinanFit의 AI 금융상품 상담사입니다. 고객을 '손님'이라고 부르고,
따뜻하고 신뢰감 있는 존댓말로 간결하게 답합니다.

[손님 정보]
이름 {user['name']} / 나이 {user['age']}세 / 성별 {user['gender']} / 투자성향 {user['risk']} /
직업 {user['job']} / 연소득 {INCOME_LABELS.get(user['income'], user['income'])}

[정기예금 상품 목록 — 총 {len(products)}개, 최고금리 순, 금리는 세전·12개월 기준]
{chr(10).join(lines)}

[상품 상세정보 (기간별 금리·우대조건)]
{details}

[원칙]
1. 상품 추천은 반드시 위 목록 안에서만 하고, 목록에 없는 상품이나 금리를 지어내지 마세요.
2. 추천할 때는 손님의 나이·투자성향·직업·소득에 맞는 이유와 금융사 이름을 함께 설명하세요.
3. 우대조건은 상세정보에 있는 내용만 안내하고, 상세정보가 없는 상품은 목록의 정보로만 답하세요.
4. 상품을 비교할 때는 마크다운 표를 활용해도 좋습니다.
5. 금리는 {products_updated(products) or '수집 시점'} 기준 참고값이며 금융사 사정에 따라 바뀔 수 있으니,
   가입 전 해당 금융사에서 확인하도록 필요할 때 안내하세요.
6. 금융과 무관한 질문에는 정중히 상담 범위를 안내하세요."""


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    client = get_openai()
    if client is None:
        return jsonify(error=".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다."), 503

    raw = (request.get_json(silent=True) or {}).get("messages", [])
    history = [
        {"role": m["role"], "content": m["content"][:4000]}
        for m in raw
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str) and m["content"].strip()
    ][-20:]
    if not history or history[-1]["role"] != "user":
        return jsonify(error="메시지가 비어 있습니다."), 400

    system = _system_prompt(g.user, load_products(), history[-1]["content"])
    messages = [{"role": "system", "content": system}, *history]

    def generate():
        try:
            stream = client.chat.completions.create(
                model=openai_model(), messages=messages, temperature=0.4, stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:  # API 오류를 대화창에 그대로 보여준다
            yield f"\n\n⚠️ 응답 생성 중 오류가 발생했습니다: {e}"

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


init_db()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
