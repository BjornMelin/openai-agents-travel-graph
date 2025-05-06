# Database Schema Documentation

## Overview

This directory contains database migration scripts for the Travel Planner system. These scripts establish the database schema in Supabase, creating tables, indexes, views, and functions necessary for the application to function.

## Schema Diagram

```
┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
│      users        │       │  travel_queries   │       │ user_preferences  │
├───────────────────┤       ├───────────────────┤       ├───────────────────┤
│ id (PK)           │       │ id (PK)           │       │ id (PK)           │
│ email             │   ┌───┤ user_id (FK)      │   ┌───┤ user_id (FK)      │
│ created_at        │   │   │ raw_query         │   │   │ travel_query_id(FK)◄──┐
│ updated_at        │◄──┘   │ destination       │◄──┘   │ preferred_airlines│   │
└───────────────────┘       │ origin            │       │ travel_class      │   │
                            │ departure_date    │       │ ...               │   │
                            │ return_date       │       └───────────────────┘   │
                            │ travelers         │                               │
                            │ budget_min        │       ┌───────────────────┐   │
                            │ budget_max        │       │   travel_plans    │   │
                            │ purpose           │       ├───────────────────┤   │
                            │ requirements      │       │ id (PK)           │   │
                            │ created_at        │       │ user_id (FK)      │   │
                            │ updated_at        │       │ travel_query_id(FK)───┘
                            └───────────────────┘       │ destination       │
                                                        │ flights           │
                                                        │ accommodation     │
                                                        │ transportation    │
                                                        │ activities        │
                                                        │ budget            │
                                                        │ overview          │
                                                        │ recommendations   │
                                                        │ alerts            │
                                                        │ metadata          │
                                                        │ created_at        │
                                                        │ updated_at        │
                                                        └───────────────────┘
```

## Tables

### users
- Stores user information
- Primary key: `id` (UUID)
- Fields: email, created_at, updated_at
- Indexes: email

### travel_queries
- Stores travel query parameters
- Primary key: `id` (UUID)
- Foreign key: `user_id` references users(id)
- Fields: raw_query, destination, origin, departure_date, return_date, travelers, budget_min, budget_max, purpose, requirements, created_at, updated_at
- Indexes: user_id, destination, departure_date/return_date

### user_preferences
- Stores user's travel preferences
- Primary key: `id` (UUID)
- Foreign keys: `user_id` references users(id), `travel_query_id` references travel_queries(id)
- Fields: Various preference fields for flights, accommodations, activities, etc.
- Indexes: user_id, travel_query_id

### travel_plans
- Stores complete travel plans
- Primary key: `id` (UUID)
- Foreign keys: `user_id` references users(id), `travel_query_id` references travel_queries(id)
- Fields: JSONB fields for various plan components (flights, accommodation, etc.), text overview, created_at, updated_at
- Indexes: user_id, travel_query_id, created_at, destination (GIN index for JSONB)

## Views

### travel_plan_summaries
- Provides a simplified view of travel plans with key information
- Joins travel_plans with travel_queries to provide date information
- Includes: id, user_id, travel_query_id, destination_name, destination_country, departure_date, return_date, travelers, total_cost, created_at, updated_at

## Functions

### search_travel_plans(search_text TEXT)
- Full-text search across travel plans
- Searches destination name, country, and overview text
- Returns matching travel_plans records

## Usage

To apply migrations, use the setup script:

```bash
python -m travel_planner.data.setup
```

Options:
- `--reset`: Drop all tables and recreate them (warning: destroys all data)
- `--test-data`: Create sample test data
- `--config`: Path to custom configuration file
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)