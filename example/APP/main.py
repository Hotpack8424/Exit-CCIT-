from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db import mongodb
from routers import site_checker
from utils import crawler
from schemas import checker
from AI import ml_preprocessing, ml_models

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(site_checker.router)
app.include_router(mongodb.router)
app.include_router(crawler.router)
app.include_router(checker.router)
app.include_router(ml_preprocessing.router)
app.include_router(ml_models.router)
