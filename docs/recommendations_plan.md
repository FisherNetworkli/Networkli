# Recommendation Roadmap & Render.com Deployment

This document captures our end‑to‑end plan for building:
- People‑matching
- Group alignment
- Event recommendations

And ensures each piece works seamlessly on Render.com.

---

## 1. Embed & Index Entities

**Goal:** Generate vector embeddings for profiles, groups, and events, then index for fast similarity

Implementation Steps:
1. Choose or train a SentenceTransformer model (e.g. `all-MiniLM-L6-v2`).
2. For each table (`profiles`, `groups`, `events`), add a JSONB column:
   - `bio_embedding`, `expertise_embedding`, etc.
   - `group_embedding`, `event_embedding`.
3. Run a one‑off script or background job to fill embeddings.
4. Enable `pgvector` extension on Supabase and add a VECTOR column (optional).
5. Create GIN/GiST index for JSONB or VECTOR.

Render.com Considerations:
- **Build Step:** Install `pgvector` in your Postgres database via Supabase dashboard.
- **Cron Job:** Use a Render Scheduled Job to run the embedding script nightly.
- **Env:** Store MODEL_PATH and SUPABASE_URL/KEY in Render Environment.

---

## 2. Materialized Views for Similarity

**Goal:** Precompute top‑N neighbors for U→U, U→Group, U→Event

Implementation Steps:
1. Write SQL functions using `embedding_column <#> other_embedding_column` or cosine SQL.
2. Create materialized views:
   - `user_to_user_similarity` (user_id, other_id, score)
   - `group_alignment`
   - `event_alignment`
3. Schedule `REFRESH MATERIALIZED VIEW CONCURRENTLY` after embeddings update.

Render.com Considerations:
- Use a Render **Scheduled Job** running `psql` commands to refresh views.
- Ensure the job service has `DATABASE_URL` from Render secrets.

---

## 3. Recommendation API Endpoints

**Goal:** Expose `GET /recommend/users`, `/groups`, `/events`

Implementation Steps:
1. In FastAPI, add routes that query the materialized views with pagination.
2. Include metadata fields (`first_name`, `last_name`, `headline`, `title`, `date`).
3. Return JSON with `id`, `score`, and display fields.

Render.com Considerations:
- Deploy the FastAPI service to Render as a Web Service.
- Set `PORT=8000` and `DATABASE_URL` in service env.

---

## 4. Analytics Aggregates

**Goal:** Compute engagement signals per group/event

Implementation Steps:
1. Write SQL to aggregate: avg. attendee embedding, RSVP counts, interaction_history counts.
2. Store results in `groups.analytics` and `events.analytics` tables or views.
3. Schedule periodic refresh.

Render.com Considerations:
- Use the same Render Scheduled Job cluster for refreshing analytics.

---

## 5. Frontend UI Integration

**Goal:** Surface recommendations in Next.js dashboard

Implementation Steps:
1. Add UI panels for "People You May Know", "Recommended Groups", "Upcoming Events".
2. Fetch from `/api/recommend/...` Next.js API routes that proxy FastAPI.
3. Display `score` as a percentage bar.

Render.com Considerations:
- Next.js deployment on Render: set `NEXT_PUBLIC_API_URL` to Render service URL.
- Ensure CORS is configured on FastAPI service.

---

## 6. Monitoring & Feedback Loop

**Goal:** Capture clicks and improve recommendations

Implementation Steps:
1. Record `interaction_history` entries of type `RECOMMENDATION_CLICK`.
2. Build a simple feedback ingestion endpoint or webhook.
3. Use collected data to retrain or fine‑tune embeddings monthly.

Render.com Considerations:
- Use Render Logs and Alerts to monitor endpoints.
- Schedule monthly retraining via Render Scheduled Job.

---

## 7. Iterate & Tune

**Goal:** Improve ranking with re‑ranking rules and A/B tests

Implementation Steps:
1. Start with simple cosine ranking.
2. Add business rules: boost same industry, friend networks, etc.
3. A/B test variations by flag in `system_settings`.

Render.com Considerations:
- Keep feature flags in `system_settings` table; update via migration.
- Deploy new versions via Render Deployment pipeline for A/B rollout.

---

*Next:* Choose which step to tackle first—let's get the embedding pipeline wired up on Render.com and Supabase! 