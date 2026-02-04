
import os
import io
import uuid
import logging
import json
import asyncio
from typing import List, Optional, Tuple
from enum import Enum

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from PIL import Image
from pdf2image import convert_from_bytes, convert_from_path
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("API_KEY", "antigravity_secret")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(header_api_key: str = Security(api_key_header)):
    if header_api_key == API_KEY:
        return header_api_key
    raise HTTPException(
        status_code=403, detail="Could not validate credentials"
    )

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Antigravity Document Processor", version="1.0.0")

# Initialize OpenAI Client
aclient = None
if OPENAI_API_KEY:
    aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OpenAI Client not initialized (Missing API Key). AI Analysis will be skipped.")

# Prompts
SYSTEM_PROMPT = """Eres un asistente experto en EXTRACCION DE DATOS. Tu tarea es analizar la imagen y generar un JSON válido.

ESTRICTAMENTE SOLO JSON.

Campos requeridos:
- MINISTERIO
- FIRMANTE
- TIPO_DOCUMENTO
- DNI
- FECHA_NACIMIENTO
- FECHA_DOCUMENTO
- ANO
- NOMBRE_APELLIDOS
- DIRECCION

Instrucciones de Formato:
1. TIPO_DOCUMENTO: Si no es claro, usa "REVISION MANUAL".
2. FECHAS: Formato DD/MM/AAAA.
3. TEXTO: MAYUSCULAS, SIN TILDES (A,E,I,O,U), Ñ->N.
4. Si un dato no se encuentra, usa null.

Responde ÚNICAMENTE con el objeto JSON."""

USER_PROMPT = "Analiza esta imagen y extrae la información solicitada en formato JSON."

import unicodedata
import re

def normalize_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    # Convert to uppercase
    text = text.upper()
    # Normalize to NFD to separate accents
    text = unicodedata.normalize('NFD', text)
    # Remove accents/diacritics
    text = "".join([c for c in text if unicodedata.category(c) != 'Mn'])
    # Specifically handle Ñ (which becomes N after NFD normalization and removing Mn)
    # Maintain forward slash (/), backward slash (\), @ and dots (.) for emails
    text = re.sub(r'[^A-Z0-9 \\/@\.]', '', text)
    return text.strip()

def extract_de_from_filename(filename: str) -> Optional[str]:
    """
    Extracts the identifier (DNI or Email) from the filename.
    """
    name = Path(filename).stem
    if "@" in name:
        # Email case: e.g. 04jmcjmc@gmail_com_15_17_35...
        parts = name.split("_")
        at_part_idx = -1
        for i, p in enumerate(parts):
            if "@" in p:
                at_part_idx = i
                break
        
        if at_part_idx != -1:
            # We take all parts from the beginning until we hit a timestamp boundary
            # Boundaries: purely numeric part with len >= 2 or 'FDV'
            end_idx = at_part_idx + 1
            while end_idx < len(parts):
                p = parts[end_idx]
                if p.isdigit() and len(p) >= 2:
                    break
                if p.upper() == "FDV":
                    break
                end_idx += 1
            
            full_email_raw = "_".join(parts[0:end_idx])
            # Reconstruct: user part as is, domain part (after @) _ becomes .
            at_pos = full_email_raw.find("@")
            user_p = full_email_raw[:at_pos]
            domain_p = full_email_raw[at_pos+1:].replace("_", ".")
            return f"{user_p}@{domain_p}"
    else:
        # DNI case: e.g. 76050910F_...
        # Returns the first block before the first underscore
        return name.split("_")[0]
    
    return None

# Data Models
class DocumentData(BaseModel):
    DE: Optional[str] = None
    MINISTERIO: Optional[str] = None
    FIRMANTE: Optional[str] = None
    TIPO_DOCUMENTO: Optional[str] = None
    DNI: Optional[str] = None
    FECHA_NACIMIENTO: Optional[str] = None
    FECHA_DOCUMENTO: Optional[str] = None
    ANO: Optional[str] = None
    NOMBRE_APELLIDOS: Optional[str] = None
    DIRECCION: Optional[str] = None

    @validator("*", pre=True)
    def normalize_fields(cls, v):
        if isinstance(v, str):
            return normalize_text(v)
        return v

class ProcessedResult(BaseModel):
    id: str
    filename: str
    status: str = "success"
    message: Optional[str] = None
    data: DocumentData
    confidence: float = 1.0

class PreprocessRequest(BaseModel):
    path: str

class PreprocessResponse(BaseModel):
    status: str
    message: Optional[str] = None
    original_file: str
    count: int
    files: List[str] = []

class ProcessRequest(BaseModel):
    path: str

async def analyze_image_with_gpt4o(image_bytes: bytes) -> DocumentData:
    """
    Analyzes an image using GPT-4o multimodal capabilities and returns a Pydantic model.
    """
    try:
        if not aclient:
            return DocumentData(TIPO_DOCUMENTO="NO_API_KEY")

        import base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        response = await aclient.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        content = response.choices[0].message.content
        logger.info(f"AI Raw Response: {content}")

        if not content:
            logger.warning("AI returned empty content. Returning empty structure.")
            return DocumentData(TIPO_DOCUMENTO="ERROR_EMPTY_RESPONSE")
        
        # Parse JSON
        data_dict = json.loads(content)
        return DocumentData(**data_dict)
            
    except Exception as e:
        logger.error(f"AI Analysis failed: {e}")
        return DocumentData(TIPO_DOCUMENTO="ERROR_ANALYSIS")

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_endpoint(request: PreprocessRequest, api_key: str = Depends(get_api_key)):
    """
    Takes a single file path, converts/splits it into PNG(s), and DELETES the original file.
    Always returns a 'files' array, even on error.
    """
    file_path = Path(request.path)
    original_name = str(file_path.name)
    
    # Initialize response with error state
    response_data = {
        "status": "error",
        "message": "",
        "original_file": original_name,
        "count": 0,
        "files": []
    }

    if not file_path.exists() or not file_path.is_file():
        response_data["message"] = f"Fichero no encontrado o ruta invalida: {request.path}"
        return response_data
    
    processed_files = []
    ext = file_path.suffix.lower()
    parent_dir = file_path.parent
    
    try:
        # 1. Handle PDFs (Split and Convert)
        if ext == ".pdf":
            # convert_from_path returns a list of PIL Images
            images = convert_from_path(str(file_path), fmt='png')
            for i, img in enumerate(images):
                output_filename = f"{file_path.stem}_page_{i+1}.png"
                output_path = parent_dir / output_filename
                img.save(output_path, "PNG")
                processed_files.append(str(output_path))
            
            logger.info(f"Preprocessed PDF: {original_name} -> {len(images)} pages. Deleting original.")
            file_path.unlink() # Delete original PDF

        # 2. Handle Images (Convert to PNG if not already)
        elif ext in [".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".jfif"]:
            img = Image.open(file_path)
            output_filename = f"{file_path.stem}.png"
            output_path = parent_dir / output_filename
            
            # Save as PNG
            img.save(output_path, "PNG")
            processed_files.append(str(output_path))
            
            logger.info(f"Preprocessed image: {original_name} -> PNG. Deleting original.")
            
            # If the output name is different from input (e.g. .jpg -> .png), delete original
            if output_path.resolve() != file_path.resolve():
                file_path.unlink()

        # 3. Handle PNGs (Already correct)
        elif ext == ".png":
            processed_files.append(str(file_path))
            logger.info(f"File is already PNG: {original_name}. No changes made.")
        
        else:
            logger.warning(f"Unsupported file type: {ext} for file {original_name}")
            response_data["message"] = f"Tipo de archivo no permitido: {ext}"
            return response_data

        # Final success response
        return {
            "status": "success",
            "message": "Procesado correctamente",
            "original_file": original_name,
            "count": len(processed_files),
            "files": processed_files
        }

    except Exception as e:
        logger.error(f"Preprocessing failed for {original_name}: {e}")
        response_data["message"] = f"Error durante el procesamiento: {str(e)}"
        return response_data

@app.post("/process", response_model=ProcessedResult)
async def process_endpoint(request: ProcessRequest, api_key: str = Depends(get_api_key)):
    """
    Analyzes a document using AI. PNGs are fully processed, while other formats 
    return basic info like 'DE' without AI analysis.
    """
    file_path = Path(request.path)
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {request.path}")
    
    # Always extract DE from filename
    de_value = extract_de_from_filename(file_path.name)
    
    # Handle non-PNG files gracefully without AI analysis
    if file_path.suffix.lower() != ".png":
        logger.info(f"Process endpoint called for non-PNG file: {file_path.name}. Returning DE only.")
        return ProcessedResult(
            id=str(uuid.uuid4()),
            filename=file_path.name,
            status="unsupported_format",
            message="El endpoint de proceso solo soporta analisis de archivos PNG. Use /preprocess primero para un analisis completo.",
            data=DocumentData(DE=de_value),
            confidence=0.0
        )
    
    logger.info(f"AI Processing PNG: {file_path.name}")
    
    try:
        with open(file_path, "rb") as f:
            img_bytes = f.read()
            
        doc_data = await analyze_image_with_gpt4o(img_bytes)
        
        # Asignar el DE extraído
        doc_data.DE = de_value
        
        # Validar si se encontraron datos relevantes (ID, DNI, etc.)
        # Excluimos DE y TIPO_DOCUMENTO para la validación de 'vacio'
        check_fields = doc_data.dict()
        important_values = [v for k, v in check_fields.items() if k not in ['DE', 'TIPO_DOCUMENTO'] and v is not None]
        
        status = "success"
        message = "Informacion extraida correctamente."
        confidence = 1.0
        
        if len(important_values) == 0:
            status = "no_data_found"
            message = "No se ha encontrado ninguna informacion relevante en el documento."
            confidence = 0.0

        return ProcessedResult(
            id=str(uuid.uuid4()),
            filename=file_path.name,
            status=status,
            message=message,
            data=doc_data,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
