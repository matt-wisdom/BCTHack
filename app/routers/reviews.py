from fastapi import APIRouter, HTTPException, Body
from app.models import SimulateReviewRequest, ReviewOutput
from app.service import rec_agent_service

router = APIRouter(prefix="/reviews", tags=["reviews"])

@router.post("/simulate", response_model=ReviewOutput)
async def simulate(request: SimulateReviewRequest = Body(...)):
    try:
        result = await rec_agent_service.simulate_review(
            request.persona_text, 
            request.product_text,
            request.mode,
            request.dataset_source
        )
        return result
    except Exception as e:
        print(f"Error in simulate review: {e}")
        raise HTTPException(status_code=500, detail=str(e))
