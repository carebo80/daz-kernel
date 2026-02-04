# app/api/plan.py
from fastapi import APIRouter
from app.core.schemas import PlanUnitRequest, PlanUnitResponse
from app.services.planning import create_plan_unit

router = APIRouter(tags=["planning"])

@router.post("/plan_unit", response_model=PlanUnitResponse)
def plan_unit(req: PlanUnitRequest):
    return create_plan_unit(req)
