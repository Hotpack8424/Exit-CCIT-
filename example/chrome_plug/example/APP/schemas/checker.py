from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()

class SiteCheckRequest(BaseModel):
    url: str

class SiteCheckResponse(BaseModel):
    url: str
    blocked: bool

class RedirectData(BaseModel):
    url: str
    redirectUrl: str
    statusCode: int