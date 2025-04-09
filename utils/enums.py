from enum import Enum

class DocumentSegments(Enum):
    TEXT = "TEXT_BLOCK"
    PAGE_HEADER = "PAGE_HEADER"
    SECTION_HEADER = "SECTION_HEADER"
    IMAGE = "IMAGE_BLOCK"
    TABLE = "TABLE_BLOCK"

class TableSegments(Enum):
    ROW = "ROW_BLOCK"
    COLUMN = "COLUMN_BLOCK"
    HEADER = "HEADER_BLOCK"
    SPAN = "SPAN_BLOCK"

class BBOX(Enum):
    X = 0
    Y = 1
    W = 2
    H = 3

class PromptRoleType(Enum):
    USER = "USER"
    SYSTEM = "SYSTEM"
    ASSISTANT = "ASSISTANT"

class LangChainRoleType(Enum):
    USER = "human"
    SYSTEM = "system"
    ASSISTANT = "ai"

class AnnotationModels(Enum):
    CLAUDE = "claude-3-haiku-20240307"
    OPENAI = "gpt-3.5-turbo"
    GOOGLE = "gemini-1.5-flash-001"
    MARITACA = "sabia-2-medium"

class ExaminationModels(Enum):
    CLAUDE = "claude-3-opus-20240229"
    OPENAI = "gpt-4-1106-preview"
    GOOGLE = "gemini-1.5-pro-001"

class ExaminationModels2(Enum):
    CLAUDE = "claude-3-5-sonnet-20240620"
    OPENAI = "gpt-4o"
    GOOGLE = "gemini-1.5-pro-001"

class EnsembleVoteMode(Enum):
    MAJORITY = "majority"
    FULL = "full"