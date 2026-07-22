# Mental Health Assessment - Backend (FastAPI)

## 1. Setup

```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env            # then fill in DATABASE_URL, CLERK_* values
```

## 2. Database

Point `DATABASE_URL` in `.env` at your Neon (or any Postgres) instance, then either:

- Let the app auto-create tables on startup (fine for local dev - see `app/main.py`), or
- Run the SQL in `../database/schema.sql` directly, or
- Use Alembic migrations for production:

```bash
alembic init alembic     # first time only, config already wired to app.database.Base
alembic revision --autogenerate -m "init"
alembic upgrade head
```

## 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Docs available at `http://localhost:8000/docs`.

## 4. The FER-2013 model

Video/Combined assessment mode uses the [`fer`](https://pypi.org/project/fer/)
Python package, which bundles its own pre-trained FER-2013 CNN. **This
means there's nothing to train or download separately** - `pip install -r
requirements.txt` is all that's needed, and `app/services/fer_service.py`
works immediately.

### Advanced/optional: using your own custom-trained model instead

If you'd rather train your own model (e.g. on a specific dataset, or to
experiment with the architecture), a full training script is provided in
`training/train_fer_model.py` - see `training/README.md` for setup
instructions. Once trained, you'd swap it in by editing the commented-out
block near the bottom of `_get_detector()` in `app/services/fer_service.py`
to load your `.h5` file instead of initializing `FER()`. This is entirely
optional - the app works fully without doing this.


## 5. Folder structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app, CORS, router wiring
│   ├── config.py            # env-driven settings
│   ├── database.py          # SQLAlchemy engine/session
│   ├── models.py            # ORM models (assessments table)
│   ├── schemas.py           # Pydantic request/response models
│   ├── auth.py               # Clerk JWT verification
│   ├── data/
│   │   └── dass21_questions.py
│   ├── services/
│   │   ├── dass_scoring.py   # DASS-21 scoring rules
│   │   ├── fer_service.py    # face detection + FER-2013 inference
│   │   └── risk_engine.py    # combines both into an overall summary
│   ├── routers/
│   │   ├── assessment.py     # /api/dass21/questions, /api/assessments
│   │   └── fer.py            # /api/fer/analyze
│   └── ml_models/            # put your trained fer2013_model.h5 here
├── requirements.txt
└── .env.example
```
