from fastapi import FastAPI

from app.api.plan import router as plan_router
from app.api.search import router as search_router
from app.api.topics import router as topics_router
from app.api.units import router as units_router
from app.api.health import router as health_router
from app.api.preview import router as preview_router

app = FastAPI(title="DaZ Kernel")

app.include_router(health_router)
app.include_router(plan_router)
app.include_router(search_router)
app.include_router(topics_router)
app.include_router(units_router)
app.include_router(preview_router)