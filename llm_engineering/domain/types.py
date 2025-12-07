from enum import StrEnum


class DataCategory(StrEnum):
    PROMPT = "prompt"
    QUERIES = "queries"

    INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"
    INSTRUCT_DATASET = "instruct_dataset"
    PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples"
    PREFERENCE_DATASET = "preference_dataset"

    SESSION = "session"

class LegalField(StrEnum):
    LAO_DONG = "Lao động"
    THUE = "Thuế"
    DAT_DAI = "Đất đai"
    DOANH_NGHIEP = "Doanh nghiệp"
    HINH_SU = "Hình sự"
    DAN_SU = "Dân sự"
    HANH_CHINH = "Hành chính"
    GIAO_DUC = "Giáo dục"
    Y_TE = "Y tế"
    TAI_CHINH = "Tài chính"
    XAY_DUNG = "Xây dựng"
    VAN_HOA = "Văn hóa"
    THUONG_MAI = "Thương mại"
    CONG_NGHE = "Công nghệ thông tin"
    TAI_NGUYEN = "Tài nguyên"

    @classmethod
    def from_url_slug(cls, slug: str) -> str:
        mapping = {
            "Lao-dong-Tien-luong": cls.LAO_DONG,
            "Thue-Phi-Le-Phi": cls.THUE,
            "Bat-dong-san": cls.DAT_DAI,
            "Doanh-nghiep": cls.DOANH_NGHIEP,
            "Hinh-su": cls.HINH_SU,
            "Dan-su": cls.DAN_SU,
            "Hanh-chinh": cls.HANH_CHINH,
            "Giao-duc": cls.GIAO_DUC,
            "Y-te": cls.Y_TE,
            "Tai-chinh-nha-nuoc": cls.TAI_CHINH,
            "Xay-dung-Do-thi": cls.XAY_DUNG,
            "Van-hoa-The-thao-Du-lich": cls.VAN_HOA,
            "Thuong-mai": cls.THUONG_MAI,
            "Cong-nghe-thong-tin": cls.CONG_NGHE,
            "Tai-nguyen-Moi-truong": cls.TAI_NGUYEN,
        }
        return mapping.get(slug, slug)


class DocumentType(StrEnum):
    LUAT = "Luật"
    NGHI_DINH = "Nghị định"
    THONG_TU = "Thông tư"
    QUYET_DINH = "Quyết định"
    NGHI_QUYET = "Nghị quyết"
    CHI_THI = "Chỉ thị"
    CONG_VAN = "Công văn"

class Role(StrEnum):
    SYSTEM = "system"
    ASSISSTANT = "assisstant"
    USER = "user"
