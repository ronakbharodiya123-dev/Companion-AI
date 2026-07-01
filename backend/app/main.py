from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
 
from app.core.config import settings
from app.core.database import Database

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts
    await Database.connect_db()
    yield
    # Runs once when the server shuts down
    await Database.close_db()
 
 
app = FastAPI(title="Companion AI (from scratch)", version="0.1.0", lifespan=lifespan)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(devices.router, prefix="/api/v1", tags=["Devices"])
 
 
@app.get("/")
async def root():
    return {"message": "Server is alive", "environment": settings.environment}
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)