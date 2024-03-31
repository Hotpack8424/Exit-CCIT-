from pydantic import BaseModel

class SiteCheckRequest(BaseModel):
    url: str

class SiteCheckResponse(BaseModel):
    url: str
    blocked: bool
    