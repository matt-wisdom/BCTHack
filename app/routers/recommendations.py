from fastapi import APIRouter, HTTPException, Body
from app.models import RecommendationRequest, RecommendationOutput
from app.service import rec_agent_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

@router.post("/generate", response_model=RecommendationOutput)
async def generate(request: RecommendationRequest = Body(...)):
    try:
        result = await rec_agent_service.generate_recommendations(
            request.persona_text,
            request.mode,
            request.dataset_source
        )
        return result
    except Exception as e:
        print(f"Error in generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
