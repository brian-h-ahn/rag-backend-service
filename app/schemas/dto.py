''' For [type correctness] purpose '''

from pydantic import BaseModel

class IngestRequest(BaseModel):
    source: str
    text: str

class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


'''
Data Transfer Object: To move data SAFELY between modules
'''