from pydantic import BaseModel

class Image_path(BaseModel) :
    img_path : str
    img_type : str