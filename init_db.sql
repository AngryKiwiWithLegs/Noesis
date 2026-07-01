CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TYPE thought_type AS ENUM (
    'identity','preference','position','question',
    'synthesis','event','contradiction'
);

CREATE TYPE thought_status AS ENUM (
    'tentative','provisional','settled'
);

CREATE TABLE IF NOT EXISTS thoughts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    title           TEXT NOT NULL,
    summary         TEXT NOT NULL,
    keywords        TEXT[],
    type            thought_type NOT NULL,
    status          thought_status NOT NULL DEFAULT 'tentative',
    confidence      FLOAT NOT NULL DEFAULT 0.2,
    source_tool     TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fact_ref        UUID REFERENCES thoughts(id),
    evolved_from    UUID REFERENCES thoughts(id),
    superseded_by   UUID REFERENCES thoughts(id),
    topic_cluster   TEXT NOT NULL DEFAULT '',
    granularity_flag TEXT,
    dedup_flag      UUID REFERENCES thoughts(id)
);

CREATE TABLE IF NOT EXISTS topic_centroids (
    cluster_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    centroid_vector VECTOR(384) NOT NULL,
    radius          FLOAT DEFAULT 0.3,
    member_count    INT DEFAULT 0,
    last_updated    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS confidence_events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thought_id  UUID REFERENCES thoughts(id),
    event_type  TEXT NOT NULL,
    delta       FLOAT NOT NULL,
    before_val  FLOAT NOT NULL,
    after_val   FLOAT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS thought_vectors (
    thought_id  UUID PRIMARY KEY REFERENCES thoughts(id) ON DELETE CASCADE,
    embedding   VECTOR(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS thought_vectors_cosine_idx
    ON thought_vectors
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
