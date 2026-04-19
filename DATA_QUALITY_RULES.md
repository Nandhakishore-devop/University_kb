# Data Quality Rules & Lifecycle Policy

This document establishes the standards for document ingestion, volunteer tracking, and data maintenance in the University Knowledge Base.

---

## 🛠️ Data Quality Rules

### 1. Posters (Event Documents)
- **Metadata Integrity**: Must include `event_date`, `venue`, and `coordinator_contact`.
- **OCR Threshold**: Minimum 80% character recognition accuracy required for automated indexing. Documents below this must be manually transcribed.
- **Tagging**: Must be tagged with `doc_type="poster"` and an appropriate `event_category`.

### 2. Attendance Records
- **Consistency**: The `student_id` must match the official university register.
- **Temporal Alignment**: Attendance timestamps must fall within official event hours.
- **Validation**: Records exceeding 8 service hours per day require a "Reason for Extension" note.

### 3. Hour Approvals
- **Audit Trail**: Every approved hour must have a `verification_id` linked to a supervisor's digital signature.
- **State Workflow**: Records remain `pending` until verified. `Rejected` records must include a rejection code (e.g., ERR_NO_SHOW, ERR_MISMATCH).

### 4. Certificate Templates
- **Standardization**: Templates must use the `{{MUSTACHE}}` syntax for dynamic fields.
- **Auditability**: Generated certificates must be hashed and the hash stored in the KB for fraud prevention.
- **Format**: Master templates must be stored in `.docx` for layout consistency.

---

## 🔄 Lifecycle: Deduplication & Archiving

### ⚡ Automated Deduplication
To prevent "Knowledge Bloat":
- **Semantic Check**: New chunks with >0.98 similarity to existing content but different metadata are flagged for "Conflict Detection".
- **Identical Metadata**: If `source_file`, `version`, and `year` match exactly, the system will perform an **Atomic Upsert** (overwriting old chunks).
- **Proactive Search**: The `AdminAgent` runs a weekly "Duplicate Sweep" to identify semantically redundant policies across departments.

### 🗄️ Archive Policy
- **Active Window**: All documents from the current and previous academic year are kept in the `active_kb` collection.
- **Auto-Archiving**:
    - Circulars older than 3 years are moved to `archive_kb`.
    - Superseded policies (where `version` is outdated) are moved to `archive_kb` immediately upon new version ingest.
- **Retention**: Attendance and hour records are retained for the duration of the student's enrollment + 1 year, then anonymized for analytics.

---

> [!IMPORTANT]
> Failure to meet Data Quality thresholds will block document visibility in the Student Search panel.
