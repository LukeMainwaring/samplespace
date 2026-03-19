from fastapi import APIRouter

from samplespace.routers import agent_router, health_router, samples_router

api_router = APIRouter()
api_router.include_router(agent_router)
api_router.include_router(health_router)
api_router.include_router(samples_router)
