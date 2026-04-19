# Event Operations Policy

This document defines the standard operating procedures for university events, including approval flows, waitlist management, and certificate eligibility.

## 1. Organizer Approval Flows
All university events must follow a multi-stage approval process before being published to the student portal:
- **Phase 1: Draft Submission**: Organizer uploads event poster and metadata (Topic, Date, Eligibility).
- **Phase 2: Administrative Review**: Admin Agent performs a conflict check against scheduled academic events and existing policies.
- **Phase 3: Formal Verification**: Department head or authorized volunteer marks the record as `verified`.
- **Phase 4: Publication**: Document status changes to `access="public"`.

## 2. Waitlist Management Rules
For over-subscribed events, the following logic applies:
- **Capacity**: Defined in the `waitlist_capacity` metadata field.
- **Priority**:
    1. Students with mandatory attendance requirements.
    2. Chronological order of registration ("First Come, First Served").
- **Automatic Promotion**: If an attendee cancels >24h before the event, the next student on the waitlist is automatically promoted and notified.

## 3. Certificate Eligibility Criteria
Certificates of participation are issued only if the following conditions are met:
- **Attendance**: Minimum 85% session duration verified via the `attendance` document type.
- **Submission**: Any post-event reflection or assessment must be marked as `verified`.
- **Timing**: Closure automation must be triggered within 7 days of event completion.

## 4. Post-Event Closure Workflow
Upon event completion, the following automated steps occur:
1. **Attendance Audit**: Aggregate all `attendance` records for the event topic.
2. **Eligibility Flagging**: Automatically tag students who meet the 85% threshold.
3. **Draft Issuance**: Generate certificate metadata for eligible participants.
4. **Archive**: Move event posters and circulars to `is_archived=True` status.
