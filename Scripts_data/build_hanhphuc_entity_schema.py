import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_JSONL = ROOT / "Data_RAG" / "retrieval" / "hanhphuc_retrieval.jsonl"
OUTPUT_DIR = ROOT / "Data_RAG" / "entities"
OUTPUT_JSONL = OUTPUT_DIR / "hanhphuc_entities.jsonl"
UPDATED_AT = "2026-04-19"


DEPARTMENT_RULES = [
    ("hỗ trợ sinh sản", "Hỗ trợ sinh sản"),
    ("hiếm muộn", "Hỗ trợ sinh sản"),
    ("sản", "Sản – Phụ khoa"),
    ("phụ khoa", "Sản – Phụ khoa"),
    ("nhi sơ sinh", "Nhi sơ sinh"),
    ("sơ sinh", "Nhi sơ sinh"),
    ("nhi", "Nhi khoa"),
    ("nội tiết", "Nội tiết"),
    ("chẩn đoán hình", "Chẩn đoán hình ảnh"),
    ("gây mê", "Gây mê hồi sức"),
    ("hồi sức", "Hồi sức"),
    ("cấp cứu", "Cấp cứu"),
    ("tai", "Tai – Mũi – Họng"),
    ("mũi", "Tai – Mũi – Họng"),
    ("họng", "Tai – Mũi – Họng"),
    ("răng", "Răng – Hàm – Mặt"),
    ("hàm", "Răng – Hàm – Mặt"),
    ("da liễu", "Da liễu"),
    ("tâm thần", "Tâm thần"),
    ("y học cổ truyền", "Y học cổ truyền – Phục hồi chức năng"),
    ("phục hồi chức năng", "Y học cổ truyền – Phục hồi chức năng"),
    ("chấn thương chỉnh hình", "Chấn thương chỉnh hình"),
    ("ngoại nhi", "Ngoại Nhi"),
    ("tạo hình", "Phẫu thuật tạo hình – Thẩm mỹ"),
    ("thẩm mỹ", "Phẫu thuật tạo hình – Thẩm mỹ"),
    ("tiết niệu", "Ngoại tiết niệu – Nam khoa"),
    ("nam khoa", "Ngoại tiết niệu – Nam khoa"),
    ("tiêu hóa", "Tiêu hóa"),
    ("ngoại tổng hợp", "Ngoại tổng hợp"),
    ("khám bệnh", "Khám bệnh"),
]

AUDIENCE_RULES = [
    ("bé trai", "bé trai"),
    ("bé gái", "bé gái"),
    ("trẻ em", "trẻ em"),
    ("trẻ", "trẻ em"),
    ("nhi", "trẻ em"),
    ("bé", "trẻ em"),
    ("sơ sinh", "trẻ sơ sinh"),
    ("phụ nữ", "phụ nữ"),
    ("nữ", "phụ nữ"),
    ("mẹ bầu", "mẹ bầu"),
    ("thai", "mẹ bầu"),
    ("sản phụ", "mẹ bầu"),
    ("người lớn", "người lớn"),
    ("người trưởng thành", "người lớn"),
    ("người cao tuổi", "người cao tuổi"),
    ("nam khoa", "nam giới"),
    ("bé trai", "nam giới"),
    ("vợ chồng", "cặp vợ chồng hiếm muộn"),
    ("hiếm muộn", "cặp vợ chồng hiếm muộn"),
]

CONDITION_KEYWORDS = [
    "dậy thì sớm",
    "chậm tăng trưởng",
    "đái tháo đường",
    "tiểu đường",
    "bệnh lý tuyến giáp",
    "tuyến giáp",
    "suy giáp",
    "nhiễm độc giáp",
    "đái tháo đường thai kỳ",
    "viêm giáp sau sinh",
    "hiếm muộn",
    "vô sinh",
    "thai kỳ nguy cơ cao",
    "thai ngoài tử cung",
    "u xơ tử cung",
    "u buồng trứng",
    "ung thư cổ tử cung",
    "ung thư vú",
    "ung thư tuyến giáp",
    "ung thư dạ dày",
    "ung thư đại trực tràng",
    "ung thư phụ khoa",
    "viêm dạ dày",
    "HP dạ dày",
    "viêm đại tràng",
    "polyp đại tràng",
    "suy hô hấp",
    "rối loạn tuần hoàn",
    "rối loạn ý thức",
    "trầm cảm",
    "lo âu",
    "mất ngủ",
    "ADHD",
    "tự kỷ",
    "OCD",
    "PTSD",
    "sa sút trí tuệ",
    "Alzheimer",
    "mụn trứng cá",
    "viêm da cơ địa",
    "nấm da",
    "rối loạn sắc tố",
    "sẹo mụn",
    "suy dinh dưỡng",
    "thừa cân",
    "béo phì",
]

PROCEDURE_KEYWORDS = [
    "nội soi",
    "siêu âm",
    "xét nghiệm",
    "chụp X-quang",
    "chụp nhũ ảnh",
    "điện tâm đồ",
    "phẫu thuật",
    "mổ sanh",
    "đỡ sanh",
    "chọc hút trứng",
    "kích thích buồng trứng",
    "chuyển phôi",
    "rã đông phôi",
    "ICSI",
    "IVF",
    "nuôi phôi",
    "tinh dịch đồ",
    "khám tiền mê",
    "gây mê",
    "tiêm chủng",
    "tiêm ngừa",
    "rTMS",
    "Laser",
    "IPL",
    "RF",
    "HIFU",
    "Thermage",
    "Mesotherapy",
    "Botox",
    "Filler",
    "Peel da",
    "đo mật độ xương",
]

KNOWN_EDUCATION_PHRASES = [
    "Đại học Y Dược TP.HCM",
    "Đại học Y Dược Thành phố Hồ Chí Minh",
    "Đại học Y Dược TP. Hồ Chí Minh",
    "Đại học Y Dược Huế",
    "Đại học Y Huế",
    "Đại học Y Dược Cần Thơ",
    "Đại học Y khoa Phạm Ngọc Thạch",
    "Đại học Y Phạm Ngọc Thạch",
    "Học viện Quân Y",
    "Đại học Y Đài Bắc",
    "Trường Đại học Y Dược Huế",
    "Trường Đại học Tây Nguyên",
]


def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def lower_text(value):
    return clean_text(value).lower()


def dedupe(items):
    seen = set()
    result = []
    for item in items:
        item = clean_text(item)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def parse_json_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return dedupe(value)
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return dedupe(parsed)
    return [text]


def read_records():
    with SOURCE_JSONL.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def strip_doctor_title(name):
    name = clean_text(name)
    original = name
    prefix_pattern = re.compile(
        r"^(?:"
        r"ThS\.?\s*BS\.?\s*CKII\.?|"
        r"ThS\.?\s*BS\.?\s*CKI\.?|"
        r"ThS\.?\s*BSNT\.?|"
        r"ThS\.?\s*BS\.?|"
        r"TS\.?\s*BS\.?|"
        r"BS\.?\s*CKII\.?|"
        r"BS\.?\s*CKI\.?|"
        r"BSNT\.?|"
        r"BS\.?|"
        r"ThS\.?|"
        r"TS\.?"
        r")\s*",
        flags=re.IGNORECASE,
    )
    previous = None
    while previous != name:
        previous = name
        name = prefix_pattern.sub("", name).strip()
    name = re.sub(r"^[.\s-]+", "", name)
    return name or original


def last_name_token(name):
    plain = strip_doctor_title(name)
    parts = plain.split()
    return parts[-1] if parts else plain


def infer_department(category, text):
    haystack = f"{category} {text}".lower()
    for needle, department in DEPARTMENT_RULES:
        if needle.lower() in haystack:
            return department
    return clean_text(category)


def infer_audience(text):
    haystack = lower_text(text)
    return dedupe(value for needle, value in AUDIENCE_RULES if needle.lower() in haystack)


def infer_keywords(text, keywords):
    haystack = lower_text(text)
    results = []
    for keyword in keywords:
        if keyword.lower() in haystack:
            results.append(keyword)
    return dedupe(results)


def infer_years_experience(text):
    matches = re.findall(r"(?:hơn|gần|trên|khoảng)?\s*(\d{1,2})\s*năm kinh nghiệm", lower_text(text))
    if not matches:
        return None
    return max(int(match) for match in matches)


def infer_education(facts):
    education = []
    for fact in facts:
        lowered = lower_text(fact)
        if not any(token in lowered for token in ["tốt nghiệp", "đại học", "học viện", "chuyên khoa", "nội trú", "thạc sĩ", "tiến sĩ", "chứng chỉ"]):
            continue

        for phrase in KNOWN_EDUCATION_PHRASES:
            if phrase.lower() in lowered:
                education.append(phrase)

        for match in re.findall(r"(Đại học [^,.;]+|Học viện [^,.;]+|Trường Đại học [^,.;]+)", fact):
            match = match.strip()
            if not any(match in item or item in match for item in education):
                education.append(match)

        for match in re.findall(r"Bác sĩ Nội trú(?: chuyên ngành)? ([^,.;]+?)(?: tại|,| và|$)", fact, flags=re.IGNORECASE):
            education.append(f"Bác sĩ Nội trú {match.strip()}")

        for match in re.findall(r"Tiến sĩ tại (Đại học [^,.;]+)", fact, flags=re.IGNORECASE):
            education.append(f"Tiến sĩ {match.strip()}")

        for match in re.findall(r"Thạc sĩ(?: [^,.;]*)?", fact, flags=re.IGNORECASE):
            education.append(match.strip())

        for match in re.findall(r"Chuyên khoa (?:I|II|1|2)(?: [^,.;]*)?", fact, flags=re.IGNORECASE):
            education.append(match.strip())

        if not education:
            education.append(fact)
    return dedupe(education)


def infer_subspecialties(department, text, conditions, procedures):
    subspecialties = []
    haystack = lower_text(text)

    if department == "Nội tiết":
        if "trẻ" in haystack or "dậy thì sớm" in haystack or "chậm tăng trưởng" in haystack:
            subspecialties.append("Nội tiết trẻ em")
        if "người lớn" in haystack or "đái tháo đường" in haystack or "tuyến giáp" in haystack:
            subspecialties.append("Nội tiết người lớn")
        if "thai kỳ" in haystack or "thai" in haystack:
            subspecialties.append("Nội tiết thai kỳ")
    elif department == "Sản – Phụ khoa":
        for phrase in ["Sản khoa", "Phụ khoa", "Thai kỳ nguy cơ cao", "Phẫu thuật nội soi phụ khoa"]:
            if phrase.lower() in haystack or any(phrase.lower() in item.lower() for item in conditions + procedures):
                subspecialties.append(phrase)
    elif department == "Nhi khoa":
        for phrase in ["Nhi tổng quát", "Tiêu hóa nhi", "Hồi sức cấp cứu nhi", "Tiêm chủng"]:
            if phrase.lower() in haystack or any(phrase.lower() in item.lower() for item in conditions + procedures):
                subspecialties.append(phrase)
    elif department == "Hỗ trợ sinh sản":
        subspecialties.extend(["Hiếm muộn", "IVF", "Hỗ trợ sinh sản"])
    elif department == "Chẩn đoán hình ảnh":
        subspecialties.append("Chẩn đoán hình ảnh")
    elif department:
        subspecialties.append(department)

    return dedupe(subspecialties)


def build_doctor_aliases(record, department, conditions, audience):
    name = clean_text(record["title"])
    short_name = strip_doctor_title(name)
    last = last_name_token(name)
    aliases = [
        short_name,
        f"bác sĩ {short_name}",
        f"bs {last}",
        f"bác sĩ {last}",
        f"bác sĩ {department.lower()}",
        f"{last} {department.lower()}",
    ]
    for item in audience[:2]:
        aliases.append(f"bác sĩ {department.lower()} {item}")
    for condition in conditions[:4]:
        aliases.append(f"bác sĩ {condition}")
        aliases.append(f"{last} {condition}")
    return dedupe(aliases)


def build_package_aliases(record, audience, recommended_for):
    title = clean_text(record["title"])
    aliases = [title, title.lower()]
    for item in audience[:3]:
        aliases.append(f"{title} {item}")
    for item in recommended_for[:4]:
        aliases.append(f"gói khám {item}")
    return dedupe(aliases)


def relationship_score(doctor, package):
    doctor_conditions = {item.lower() for item in doctor.get("conditions", [])}
    package_conditions = {item.lower() for item in package.get("recommended_for", [])}
    doctor_procedures = {item.lower() for item in doctor.get("procedures", [])}
    package_procedures = {item.lower() for item in package.get("includes", [])}
    doctor_audience = {item.lower() for item in doctor.get("audience", [])}
    package_audience = {item.lower() for item in package.get("audience", [])}

    score = 0
    score += 4 * len(doctor_conditions & package_conditions)
    score += 2 * len(doctor_procedures & package_procedures)
    score += len(doctor_audience & package_audience)

    package_text = " ".join(base_terms(package)).lower()
    if doctor.get("department") == "Nội tiết" and "dậy thì sớm" in package_text:
        score += 6
    if doctor.get("department") == "Hỗ trợ sinh sản" and package.get("category") == "Hỗ trợ sinh sản":
        score += 5
    if doctor.get("department") == "Sản – Phụ khoa" and package.get("category") in {"Sức khỏe phụ nữ", "Khám thai", "Sinh"}:
        score += 4
    if doctor.get("department") == "Sản – Phụ khoa" and package.get("category") == "Hỗ trợ sinh sản":
        score += 2
    if doctor.get("department") in {"Nhi khoa", "Nhi sơ sinh"} and package.get("category") in {"Khám trẻ em", "Tiêm chủng"}:
        score += 2
    return score


def build_search_text(*parts):
    values = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, list):
            values.extend(clean_text(item) for item in part if clean_text(item))
        else:
            text = clean_text(part)
            if text:
                values.append(text)
    return ", ".join(dedupe(values))


def base_terms(entity):
    terms = []
    for key in [
        "department",
        "category",
        "canonical_name",
        "summary_text",
    ]:
        value = entity.get(key)
        if value:
            terms.append(value)
    for key in [
        "aliases",
        "subspecialties",
        "audience",
        "conditions",
        "procedures",
        "recommended_for",
        "includes",
    ]:
        terms.extend(entity.get(key) or [])
    return dedupe(terms)


def build_doctor_entity(record):
    facts = parse_json_list(record.get("facts_json"))
    text = build_search_text(record.get("title"), record.get("category"), record.get("summary"), facts)
    department = infer_department(record.get("category"), text)
    conditions = infer_keywords(text, CONDITION_KEYWORDS)
    procedures = infer_keywords(text, PROCEDURE_KEYWORDS)
    audience = infer_audience(text)
    subspecialties = infer_subspecialties(department, text, conditions, procedures)
    education = infer_education(facts)

    return {
        "entity_id": record["id"],
        "entity_type": "doctor",
        "canonical_name": clean_text(record["title"]),
        "aliases": build_doctor_aliases(record, department, conditions, audience),
        "department": department,
        "subspecialties": subspecialties,
        "audience": audience,
        "conditions": conditions,
        "procedures": procedures,
        "years_experience": infer_years_experience(text),
        "education": education,
        "source_url": record.get("source_url"),
        "summary_text": clean_text(record.get("summary")),
        "search_text": "",
        "related_packages": [],
        "updated_at": UPDATED_AT,
    }


def build_package_entity(record):
    services = parse_json_list(record.get("services_json"))
    preparation = parse_json_list(record.get("preparation_json"))
    terms = parse_json_list(record.get("terms_json"))
    text = build_search_text(record.get("title"), record.get("category"), record.get("summary"), services, preparation, terms)
    conditions = infer_keywords(text, CONDITION_KEYWORDS)
    procedures = infer_keywords(text, PROCEDURE_KEYWORDS)
    audience = infer_audience(text)
    recommended_for = dedupe(conditions)
    if "dậy thì sớm" in lower_text(text):
        if "bé trai" in lower_text(text):
            recommended_for.extend(["nghi ngờ dậy thì sớm ở bé trai", "mọc lông sớm", "vỡ giọng sớm"])
        elif "bé gái" in lower_text(text):
            recommended_for.extend(["nghi ngờ dậy thì sớm ở bé gái", "phát triển ngực sớm", "có kinh sớm"])
        else:
            recommended_for.append("nghi ngờ dậy thì sớm")

    return {
        "entity_id": record["id"],
        "entity_type": "package",
        "canonical_name": clean_text(record["title"]),
        "aliases": build_package_aliases(record, audience, recommended_for),
        "category": clean_text(record["category"]),
        "audience": audience,
        "recommended_for": dedupe(recommended_for),
        "includes": services,
        "preparation": preparation,
        "terms": terms,
        "price_vnd": record.get("price_vnd"),
        "related_doctors": [],
        "search_text": "",
        "source_url": record.get("source_url"),
        "updated_at": UPDATED_AT,
    }


def assign_relationships(doctors, packages):
    for doctor in doctors:
        scored = []
        for package in packages:
            score = relationship_score(doctor, package)
            if score >= 3:
                scored.append((score, package["entity_id"]))
        doctor["related_packages"] = [entity_id for _, entity_id in sorted(scored, reverse=True)[:8]]

    for package in packages:
        scored = []
        for doctor in doctors:
            score = relationship_score(doctor, package)
            if score >= 4:
                scored.append((score, doctor["entity_id"]))
        package["related_doctors"] = [entity_id for _, entity_id in sorted(scored, reverse=True)[:5]]


def finalize_search_text(entities):
    for entity in entities:
        if entity["entity_type"] == "doctor":
            entity["search_text"] = build_search_text(
                entity["canonical_name"],
                entity["aliases"],
                entity["department"],
                entity["subspecialties"],
                entity["audience"],
                entity["conditions"],
                entity["procedures"],
                entity["education"],
                entity["summary_text"],
            )
        else:
            entity["search_text"] = build_search_text(
                entity["canonical_name"],
                entity["aliases"],
                entity["category"],
                entity["audience"],
                entity["recommended_for"],
                entity["includes"],
                entity["preparation"],
                entity["terms"],
            )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = read_records()
    doctors = [build_doctor_entity(record) for record in records if record.get("entity_type") == "doctor"]
    packages = [build_package_entity(record) for record in records if record.get("entity_type") == "package"]

    assign_relationships(doctors, packages)
    entities = doctors + packages
    finalize_search_text(entities)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for entity in entities:
            handle.write(json.dumps(entity, ensure_ascii=False) + "\n")

    stats = {
        "source": str(SOURCE_JSONL),
        "output": str(OUTPUT_JSONL),
        "doctors": len(doctors),
        "packages": len(packages),
        "entities": len(entities),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
