-- -- Enable the pgvector extension
-- create extension if not exists vector;

-- -- Create the documentation chunks table
-- create table site_pages (
--     id bigserial primary key,
--     url varchar not null,
--     chunk_number integer not null,
--     title varchar not null,
--     summary varchar not null,
--     content text not null,  -- Added content column
--     metadata jsonb not null default '{}'::jsonb,  -- Added metadata column
--     embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
--     created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
--     -- Add a unique constraint to prevent duplicate chunks for the same URL
--     unique(url, chunk_number)
-- );

-- -- Create an index for better vector similarity search performance
-- create index on site_pages using ivfflat (embedding vector_cosine_ops);

-- -- Create an index on metadata for faster filtering
-- create index idx_site_pages_metadata on site_pages using gin (metadata);

-- -- Create a function to search for documentation chunks
-- create function match_site_pages (
--   query_embedding vector(1536),
--   match_count int default 10,
--   filter jsonb DEFAULT '{}'::jsonb
-- ) returns table (
--   id bigint,
--   url varchar,
--   chunk_number integer,
--   title varchar,
--   summary varchar,
--   content text,
--   metadata jsonb,
--   similarity float
-- )
-- language plpgsql
-- as $$
-- #variable_conflict use_column
-- begin
--   return query
--   select
--     id,
--     url,
--     chunk_number,
--     title,
--     summary,
--     content,
--     metadata,
--     1 - (site_pages.embedding <=> query_embedding) as similarity
--   from site_pages
--   where metadata @> filter
--   order by site_pages.embedding <=> query_embedding
--   limit match_count;
-- end;
-- $$;

-- -- Everything above will work for any PostgreSQL database. The below commands are for Supabase security

-- -- Enable RLS on the table
-- alter table site_pages enable row level security;

-- -- Create a policy that allows anyone to read
-- create policy "Allow public read access"
--   on site_pages
--   for select
--   to public
--   using (true);

-- Enable pgvector
create extension if not exists vector;

-- Create the documentation chunks table (embedding = 768 dims)
create table site_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(768),  -- Ollama nomic-embed-text = 768 dims
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,

    unique(url, chunk_number)
);

-- Vector index (cosine). Tune lists as needed.
create index if not exists site_pages_embedding_idx
  on site_pages using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Metadata index
create index if not exists idx_site_pages_metadata
  on site_pages using gin (metadata);

-- Search function (note query_embedding is now vector(768))
create or replace function match_site_pages (
  query_embedding vector(768),
  match_count int default 10,
  filter jsonb default '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
  from site_pages
  where metadata @> filter
  order by site_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Supabase RLS
alter table site_pages enable row level security;

create policy "Allow public read access"
  on site_pages
  for select
  to public
  using (true);



'''
-- 1) Drop old vector index (name may vary; this drops ours if it exists)
drop index if exists site_pages_embedding_idx;

-- (Optional helper: list any existing indexes if you used an unnamed one previously)
-- select indexname from pg_indexes where tablename = 'site_pages';

-- 2) Change the column to 768 dims
alter table site_pages
  alter column embedding type vector(768);

-- 3) Recreate the IVFFLAT index with cosine
create index if not exists site_pages_embedding_idx
  on site_pages using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 4) Update function signature to 768
create or replace function match_site_pages (
  query_embedding vector(768),
  match_count int default 10,
  filter jsonb default '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
  from site_pages
  where metadata @> filter
  order by site_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;
'''