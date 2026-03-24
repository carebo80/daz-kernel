--
-- PostgreSQL database dump
--

\restrict PMSUdhGkqZSxruCAtxtw408cAcRzYSdq0OSffzxCr1Rl6OMyjvhfl7lEeuAgC2i

-- Dumped from database version 16.13 (Ubuntu 16.13-0ubuntu0.24.04.1)
-- Dumped by pg_dump version 16.13 (Ubuntu 16.13-0ubuntu0.24.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;


--
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: attempts; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.attempts (
    id bigint NOT NULL,
    user_id bigint,
    competence text,
    task_json jsonb,
    answer_text text,
    score double precision,
    feedback text,
    citations jsonb,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.attempts OWNER TO daz;

--
-- Name: attempts_id_seq; Type: SEQUENCE; Schema: public; Owner: daz
--

CREATE SEQUENCE public.attempts_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.attempts_id_seq OWNER TO daz;

--
-- Name: attempts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: daz
--

ALTER SEQUENCE public.attempts_id_seq OWNED BY public.attempts.id;


--
-- Name: cefr_descriptor; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.cefr_descriptor (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    source_id uuid NOT NULL,
    scale_id uuid,
    level text NOT NULL,
    descriptor text NOT NULL,
    descriptor_short text,
    context_notes text,
    is_negative boolean DEFAULT false NOT NULL,
    sort_order integer DEFAULT 0 NOT NULL,
    meta jsonb DEFAULT '{}'::jsonb NOT NULL,
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.cefr_descriptor OWNER TO daz;

--
-- Name: cefr_descriptor_tags; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.cefr_descriptor_tags (
    descriptor_id uuid NOT NULL,
    tag_id uuid NOT NULL,
    weight real DEFAULT 1.0 NOT NULL,
    meta jsonb DEFAULT '{}'::jsonb NOT NULL
);


ALTER TABLE public.cefr_descriptor_tags OWNER TO daz;

--
-- Name: cefr_scale; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.cefr_scale (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    source_id uuid NOT NULL,
    code text NOT NULL,
    title text NOT NULL,
    skill text NOT NULL,
    domain text,
    parent_id uuid,
    sort_order integer DEFAULT 0 NOT NULL,
    meta jsonb DEFAULT '{}'::jsonb NOT NULL,
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.cefr_scale OWNER TO daz;

--
-- Name: cefr_source; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.cefr_source (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    code text NOT NULL,
    title text NOT NULL,
    publisher text,
    year integer,
    language text DEFAULT 'de'::text NOT NULL,
    uri text,
    meta jsonb DEFAULT '{}'::jsonb NOT NULL,
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.cefr_source OWNER TO daz;

--
-- Name: doc_chunks; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.doc_chunks (
    id bigint NOT NULL,
    source text NOT NULL,
    page integer,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    embedding public.vector(384),
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.doc_chunks OWNER TO daz;

--
-- Name: doc_chunks_id_seq; Type: SEQUENCE; Schema: public; Owner: daz
--

CREATE SEQUENCE public.doc_chunks_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.doc_chunks_id_seq OWNER TO daz;

--
-- Name: doc_chunks_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: daz
--

ALTER SEQUENCE public.doc_chunks_id_seq OWNED BY public.doc_chunks.id;


--
-- Name: generated_outputs; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.generated_outputs (
    id bigint NOT NULL,
    kind text NOT NULL,
    input jsonb NOT NULL,
    prompt text NOT NULL,
    output text NOT NULL,
    citations jsonb,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.generated_outputs OWNER TO daz;

--
-- Name: generated_outputs_id_seq; Type: SEQUENCE; Schema: public; Owner: daz
--

CREATE SEQUENCE public.generated_outputs_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.generated_outputs_id_seq OWNER TO daz;

--
-- Name: generated_outputs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: daz
--

ALTER SEQUENCE public.generated_outputs_id_seq OWNED BY public.generated_outputs.id;


--
-- Name: p_assets; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_assets (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    type text NOT NULL,
    uri text NOT NULL,
    mime text,
    width integer,
    height integer,
    sha256 text,
    source text,
    prompt text,
    model text,
    meta jsonb DEFAULT '{}'::jsonb NOT NULL,
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.p_assets OWNER TO daz;

--
-- Name: p_levels; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_levels (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    code text NOT NULL,
    title text NOT NULL,
    constraints jsonb DEFAULT '{}'::jsonb NOT NULL
);


ALTER TABLE public.p_levels OWNER TO daz;

--
-- Name: p_phase_models; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_phase_models (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    code text NOT NULL,
    title text NOT NULL,
    description text DEFAULT ''::text,
    schema jsonb DEFAULT '{}'::jsonb NOT NULL,
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.p_phase_models OWNER TO daz;

--
-- Name: p_skills; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_skills (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    code text NOT NULL,
    title text NOT NULL,
    description text DEFAULT ''::text NOT NULL,
    tags text[] DEFAULT '{}'::text[] NOT NULL
);


ALTER TABLE public.p_skills OWNER TO daz;

--
-- Name: p_tags; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_tags (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    type text NOT NULL,
    code text NOT NULL,
    title text NOT NULL,
    parent_id uuid,
    is_active boolean DEFAULT true NOT NULL,
    meta jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.p_tags OWNER TO daz;

--
-- Name: p_topics; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_topics (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    slug text NOT NULL,
    title text NOT NULL
);


ALTER TABLE public.p_topics OWNER TO daz;

--
-- Name: p_unit_citations; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_unit_citations (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    unit_id uuid NOT NULL,
    chunk_id bigint NOT NULL,
    score double precision DEFAULT 0.0 NOT NULL,
    quote text DEFAULT ''::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.p_unit_citations OWNER TO daz;

--
-- Name: p_unit_skills; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_unit_skills (
    unit_id uuid NOT NULL,
    skill_id uuid NOT NULL
);


ALTER TABLE public.p_unit_skills OWNER TO daz;

--
-- Name: p_unit_tags; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_unit_tags (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    unit_id uuid NOT NULL,
    tag_id uuid NOT NULL,
    role text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.p_unit_tags OWNER TO daz;

--
-- Name: p_units; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_units (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    level_id uuid NOT NULL,
    topic_id uuid,
    time_start text DEFAULT ''::text NOT NULL,
    time_end text DEFAULT ''::text NOT NULL,
    strong_group boolean DEFAULT false NOT NULL,
    title text DEFAULT ''::text NOT NULL,
    notes text DEFAULT ''::text NOT NULL,
    plan jsonb DEFAULT '{}'::jsonb NOT NULL,
    language_support jsonb DEFAULT '{}'::jsonb NOT NULL,
    phase_model_id uuid
);


ALTER TABLE public.p_units OWNER TO daz;

--
-- Name: p_vocabulary; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_vocabulary (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    lemma text NOT NULL,
    pos text,
    article text,
    plural text,
    level text,
    definition text,
    example text,
    image_path text,
    audio_path text,
    meta jsonb DEFAULT '{}'::jsonb NOT NULL,
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    image_prompt text
);


ALTER TABLE public.p_vocabulary OWNER TO daz;

--
-- Name: p_vocabulary_assets; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_vocabulary_assets (
    vocabulary_id uuid NOT NULL,
    asset_id uuid NOT NULL,
    role text DEFAULT 'main'::text NOT NULL,
    sort_order integer DEFAULT 0 NOT NULL
);


ALTER TABLE public.p_vocabulary_assets OWNER TO daz;

--
-- Name: p_vocabulary_tags; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.p_vocabulary_tags (
    vocabulary_id uuid NOT NULL,
    tag_id uuid NOT NULL
);


ALTER TABLE public.p_vocabulary_tags OWNER TO daz;

--
-- Name: skills; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.skills (
    id bigint NOT NULL,
    user_id bigint NOT NULL,
    competence text NOT NULL,
    level double precision DEFAULT 0.0 NOT NULL,
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.skills OWNER TO daz;

--
-- Name: skills_id_seq; Type: SEQUENCE; Schema: public; Owner: daz
--

CREATE SEQUENCE public.skills_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.skills_id_seq OWNER TO daz;

--
-- Name: skills_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: daz
--

ALTER SEQUENCE public.skills_id_seq OWNED BY public.skills.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: daz
--

CREATE TABLE public.users (
    id bigint NOT NULL,
    username text NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.users OWNER TO daz;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: daz
--

CREATE SEQUENCE public.users_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO daz;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: daz
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: attempts id; Type: DEFAULT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.attempts ALTER COLUMN id SET DEFAULT nextval('public.attempts_id_seq'::regclass);


--
-- Name: doc_chunks id; Type: DEFAULT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.doc_chunks ALTER COLUMN id SET DEFAULT nextval('public.doc_chunks_id_seq'::regclass);


--
-- Name: generated_outputs id; Type: DEFAULT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.generated_outputs ALTER COLUMN id SET DEFAULT nextval('public.generated_outputs_id_seq'::regclass);


--
-- Name: skills id; Type: DEFAULT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.skills ALTER COLUMN id SET DEFAULT nextval('public.skills_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Name: attempts attempts_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.attempts
    ADD CONSTRAINT attempts_pkey PRIMARY KEY (id);


--
-- Name: cefr_descriptor cefr_descriptor_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.cefr_descriptor
    ADD CONSTRAINT cefr_descriptor_pkey PRIMARY KEY (id);


--
-- Name: cefr_descriptor_tags cefr_descriptor_tags_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.cefr_descriptor_tags
    ADD CONSTRAINT cefr_descriptor_tags_pkey PRIMARY KEY (descriptor_id, tag_id);


--
-- Name: cefr_scale cefr_scale_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.cefr_scale
    ADD CONSTRAINT cefr_scale_pkey PRIMARY KEY (id);


--
-- Name: cefr_scale cefr_scale_source_id_code_key; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.cefr_scale
    ADD CONSTRAINT cefr_scale_source_id_code_key UNIQUE (source_id, code);


--
-- Name: cefr_source cefr_source_code_key; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.cefr_source
    ADD CONSTRAINT cefr_source_code_key UNIQUE (code);


--
-- Name: cefr_source cefr_source_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.cefr_source
    ADD CONSTRAINT cefr_source_pkey PRIMARY KEY (id);


--
-- Name: doc_chunks doc_chunks_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.doc_chunks
    ADD CONSTRAINT doc_chunks_pkey PRIMARY KEY (id);


--
-- Name: generated_outputs generated_outputs_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.generated_outputs
    ADD CONSTRAINT generated_outputs_pkey PRIMARY KEY (id);


--
-- Name: p_assets p_assets_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_assets
    ADD CONSTRAINT p_assets_pkey PRIMARY KEY (id);


--
-- Name: p_levels p_levels_code_key; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_levels
    ADD CONSTRAINT p_levels_code_key UNIQUE (code);


--
-- Name: p_levels p_levels_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_levels
    ADD CONSTRAINT p_levels_pkey PRIMARY KEY (id);


--
-- Name: p_phase_models p_phase_models_code_key; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_phase_models
    ADD CONSTRAINT p_phase_models_code_key UNIQUE (code);


--
-- Name: p_phase_models p_phase_models_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_phase_models
    ADD CONSTRAINT p_phase_models_pkey PRIMARY KEY (id);


--
-- Name: p_skills p_skills_code_key; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_skills
    ADD CONSTRAINT p_skills_code_key UNIQUE (code);


--
-- Name: p_skills p_skills_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_skills
    ADD CONSTRAINT p_skills_pkey PRIMARY KEY (id);


--
-- Name: p_tags p_tags_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_tags
    ADD CONSTRAINT p_tags_pkey PRIMARY KEY (id);


--
-- Name: p_tags p_tags_type_code_key; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_tags
    ADD CONSTRAINT p_tags_type_code_key UNIQUE (type, code);


--
-- Name: p_topics p_topics_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_topics
    ADD CONSTRAINT p_topics_pkey PRIMARY KEY (id);


--
-- Name: p_topics p_topics_slug_key; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_topics
    ADD CONSTRAINT p_topics_slug_key UNIQUE (slug);


--
-- Name: p_unit_citations p_unit_citations_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_unit_citations
    ADD CONSTRAINT p_unit_citations_pkey PRIMARY KEY (id);


--
-- Name: p_unit_skills p_unit_skills_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_unit_skills
    ADD CONSTRAINT p_unit_skills_pkey PRIMARY KEY (unit_id, skill_id);


--
-- Name: p_unit_tags p_unit_tags_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_unit_tags
    ADD CONSTRAINT p_unit_tags_pkey PRIMARY KEY (id);


--
-- Name: p_unit_tags p_unit_tags_unit_id_tag_id_key; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_unit_tags
    ADD CONSTRAINT p_unit_tags_unit_id_tag_id_key UNIQUE (unit_id, tag_id);


--
-- Name: p_units p_units_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_units
    ADD CONSTRAINT p_units_pkey PRIMARY KEY (id);


--
-- Name: p_vocabulary_assets p_vocabulary_assets_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_vocabulary_assets
    ADD CONSTRAINT p_vocabulary_assets_pkey PRIMARY KEY (vocabulary_id, asset_id);


--
-- Name: p_vocabulary p_vocabulary_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_vocabulary
    ADD CONSTRAINT p_vocabulary_pkey PRIMARY KEY (id);


--
-- Name: p_vocabulary_tags p_vocabulary_tags_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_vocabulary_tags
    ADD CONSTRAINT p_vocabulary_tags_pkey PRIMARY KEY (vocabulary_id, tag_id);


--
-- Name: skills skills_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.skills
    ADD CONSTRAINT skills_pkey PRIMARY KEY (id);


--
-- Name: skills skills_user_id_competence_key; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.skills
    ADD CONSTRAINT skills_user_id_competence_key UNIQUE (user_id, competence);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- Name: idx_p_unit_citations_chunk; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX idx_p_unit_citations_chunk ON public.p_unit_citations USING btree (chunk_id);


--
-- Name: idx_p_unit_citations_unit; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX idx_p_unit_citations_unit ON public.p_unit_citations USING btree (unit_id);


--
-- Name: idx_p_units_level; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX idx_p_units_level ON public.p_units USING btree (level_id);


--
-- Name: idx_p_units_topic; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX idx_p_units_topic ON public.p_units USING btree (topic_id);


--
-- Name: ix_assets_sha; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_assets_sha ON public.p_assets USING btree (sha256);


--
-- Name: ix_assets_type; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_assets_type ON public.p_assets USING btree (type);


--
-- Name: ix_cefr_descriptor_level; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_cefr_descriptor_level ON public.cefr_descriptor USING btree (level);


--
-- Name: ix_cefr_descriptor_scale; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_cefr_descriptor_scale ON public.cefr_descriptor USING btree (scale_id);


--
-- Name: ix_cefr_descriptor_source; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_cefr_descriptor_source ON public.cefr_descriptor USING btree (source_id);


--
-- Name: ix_cefr_scale_skill; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_cefr_scale_skill ON public.cefr_scale USING btree (skill);


--
-- Name: ix_doc_chunks_embedding_ivfflat; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_doc_chunks_embedding_ivfflat ON public.doc_chunks USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100');


--
-- Name: ix_p_assets_created_at; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_assets_created_at ON public.p_assets USING btree (created_at DESC);


--
-- Name: ix_p_assets_type; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_assets_type ON public.p_assets USING btree (type);


--
-- Name: ix_p_assets_type_active; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_assets_type_active ON public.p_assets USING btree (type, is_active);


--
-- Name: ix_p_tags_parent; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_tags_parent ON public.p_tags USING btree (parent_id);


--
-- Name: ix_p_tags_parent_type; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_tags_parent_type ON public.p_tags USING btree (parent_id, type);


--
-- Name: ix_p_tags_type; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_tags_type ON public.p_tags USING btree (type);


--
-- Name: ix_p_tags_type_title; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_tags_type_title ON public.p_tags USING btree (type, title);


--
-- Name: ix_p_unit_tags_tag; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_unit_tags_tag ON public.p_unit_tags USING btree (tag_id);


--
-- Name: ix_p_unit_tags_unit; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_unit_tags_unit ON public.p_unit_tags USING btree (unit_id);


--
-- Name: ix_p_vocabulary_assets_asset; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_vocabulary_assets_asset ON public.p_vocabulary_assets USING btree (asset_id);


--
-- Name: ix_p_vocabulary_assets_vocab; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_vocabulary_assets_vocab ON public.p_vocabulary_assets USING btree (vocabulary_id);


--
-- Name: ix_p_vocabulary_tags_tag; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_vocabulary_tags_tag ON public.p_vocabulary_tags USING btree (tag_id);


--
-- Name: ix_p_vocabulary_tags_vocab; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_p_vocabulary_tags_vocab ON public.p_vocabulary_tags USING btree (vocabulary_id);


--
-- Name: ix_vocab_active; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_vocab_active ON public.p_vocabulary USING btree (is_active);


--
-- Name: ix_vocab_assets_vocab; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_vocab_assets_vocab ON public.p_vocabulary_assets USING btree (vocabulary_id);


--
-- Name: ix_vocab_lemma; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_vocab_lemma ON public.p_vocabulary USING btree (lemma);


--
-- Name: ix_vocab_level; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_vocab_level ON public.p_vocabulary USING btree (level);


--
-- Name: ix_vocab_pos; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_vocab_pos ON public.p_vocabulary USING btree (pos);


--
-- Name: ix_vocab_tags_tag; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_vocab_tags_tag ON public.p_vocabulary_tags USING btree (tag_id);


--
-- Name: ix_vocab_tags_vocab; Type: INDEX; Schema: public; Owner: daz
--

CREATE INDEX ix_vocab_tags_vocab ON public.p_vocabulary_tags USING btree (vocabulary_id);


--
-- Name: uq_doc_chunks_source_page_chunk; Type: INDEX; Schema: public; Owner: daz
--

CREATE UNIQUE INDEX uq_doc_chunks_source_page_chunk ON public.doc_chunks USING btree (source, page, chunk_index);


--
-- Name: ux_vocab_lemma_level; Type: INDEX; Schema: public; Owner: daz
--

CREATE UNIQUE INDEX ux_vocab_lemma_level ON public.p_vocabulary USING btree (lower(lemma), COALESCE(level, ''::text));


--
-- Name: attempts attempts_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.attempts
    ADD CONSTRAINT attempts_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: cefr_descriptor cefr_descriptor_scale_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.cefr_descriptor
    ADD CONSTRAINT cefr_descriptor_scale_id_fkey FOREIGN KEY (scale_id) REFERENCES public.cefr_scale(id) ON DELETE SET NULL;


--
-- Name: cefr_descriptor cefr_descriptor_source_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.cefr_descriptor
    ADD CONSTRAINT cefr_descriptor_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.cefr_source(id) ON DELETE CASCADE;


--
-- Name: cefr_descriptor_tags cefr_descriptor_tags_descriptor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.cefr_descriptor_tags
    ADD CONSTRAINT cefr_descriptor_tags_descriptor_id_fkey FOREIGN KEY (descriptor_id) REFERENCES public.cefr_descriptor(id) ON DELETE CASCADE;


--
-- Name: cefr_descriptor_tags cefr_descriptor_tags_tag_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.cefr_descriptor_tags
    ADD CONSTRAINT cefr_descriptor_tags_tag_id_fkey FOREIGN KEY (tag_id) REFERENCES public.p_tags(id) ON DELETE CASCADE;


--
-- Name: cefr_scale cefr_scale_parent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.cefr_scale
    ADD CONSTRAINT cefr_scale_parent_id_fkey FOREIGN KEY (parent_id) REFERENCES public.cefr_scale(id);


--
-- Name: cefr_scale cefr_scale_source_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.cefr_scale
    ADD CONSTRAINT cefr_scale_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.cefr_source(id) ON DELETE CASCADE;


--
-- Name: p_tags p_tags_parent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_tags
    ADD CONSTRAINT p_tags_parent_id_fkey FOREIGN KEY (parent_id) REFERENCES public.p_tags(id) ON DELETE SET NULL;


--
-- Name: p_unit_citations p_unit_citations_chunk_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_unit_citations
    ADD CONSTRAINT p_unit_citations_chunk_id_fkey FOREIGN KEY (chunk_id) REFERENCES public.doc_chunks(id);


--
-- Name: p_unit_citations p_unit_citations_unit_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_unit_citations
    ADD CONSTRAINT p_unit_citations_unit_id_fkey FOREIGN KEY (unit_id) REFERENCES public.p_units(id) ON DELETE CASCADE;


--
-- Name: p_unit_skills p_unit_skills_skill_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_unit_skills
    ADD CONSTRAINT p_unit_skills_skill_id_fkey FOREIGN KEY (skill_id) REFERENCES public.p_skills(id);


--
-- Name: p_unit_skills p_unit_skills_unit_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_unit_skills
    ADD CONSTRAINT p_unit_skills_unit_id_fkey FOREIGN KEY (unit_id) REFERENCES public.p_units(id) ON DELETE CASCADE;


--
-- Name: p_unit_tags p_unit_tags_tag_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_unit_tags
    ADD CONSTRAINT p_unit_tags_tag_id_fkey FOREIGN KEY (tag_id) REFERENCES public.p_tags(id) ON DELETE CASCADE;


--
-- Name: p_unit_tags p_unit_tags_unit_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_unit_tags
    ADD CONSTRAINT p_unit_tags_unit_id_fkey FOREIGN KEY (unit_id) REFERENCES public.p_units(id) ON DELETE CASCADE;


--
-- Name: p_units p_units_level_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_units
    ADD CONSTRAINT p_units_level_id_fkey FOREIGN KEY (level_id) REFERENCES public.p_levels(id);


--
-- Name: p_units p_units_phase_model_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_units
    ADD CONSTRAINT p_units_phase_model_id_fkey FOREIGN KEY (phase_model_id) REFERENCES public.p_phase_models(id);


--
-- Name: p_units p_units_topic_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_units
    ADD CONSTRAINT p_units_topic_id_fkey FOREIGN KEY (topic_id) REFERENCES public.p_topics(id);


--
-- Name: p_vocabulary_assets p_vocabulary_assets_asset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_vocabulary_assets
    ADD CONSTRAINT p_vocabulary_assets_asset_id_fkey FOREIGN KEY (asset_id) REFERENCES public.p_assets(id) ON DELETE CASCADE;


--
-- Name: p_vocabulary_assets p_vocabulary_assets_vocabulary_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_vocabulary_assets
    ADD CONSTRAINT p_vocabulary_assets_vocabulary_id_fkey FOREIGN KEY (vocabulary_id) REFERENCES public.p_vocabulary(id) ON DELETE CASCADE;


--
-- Name: p_vocabulary_tags p_vocabulary_tags_tag_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_vocabulary_tags
    ADD CONSTRAINT p_vocabulary_tags_tag_id_fkey FOREIGN KEY (tag_id) REFERENCES public.p_tags(id) ON DELETE CASCADE;


--
-- Name: p_vocabulary_tags p_vocabulary_tags_vocabulary_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.p_vocabulary_tags
    ADD CONSTRAINT p_vocabulary_tags_vocabulary_id_fkey FOREIGN KEY (vocabulary_id) REFERENCES public.p_vocabulary(id) ON DELETE CASCADE;


--
-- Name: skills skills_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: daz
--

ALTER TABLE ONLY public.skills
    ADD CONSTRAINT skills_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict PMSUdhGkqZSxruCAtxtw408cAcRzYSdq0OSffzxCr1Rl6OMyjvhfl7lEeuAgC2i

