-- Initial schema for the travel planner system

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create an index on the users email for quicker lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Travel queries table
CREATE TABLE IF NOT EXISTS travel_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    raw_query TEXT NOT NULL,
    destination TEXT,
    origin TEXT,
    departure_date DATE,
    return_date DATE,
    travelers INTEGER DEFAULT 1,
    budget_min NUMERIC,
    budget_max NUMERIC,
    purpose TEXT,
    requirements JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_travel_queries_user_id ON travel_queries(user_id);
CREATE INDEX IF NOT EXISTS idx_travel_queries_destination ON travel_queries(destination);
CREATE INDEX IF NOT EXISTS idx_travel_queries_dates ON travel_queries(departure_date, return_date);

-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    travel_query_id UUID REFERENCES travel_queries(id),
    preferred_airlines JSONB,
    travel_class TEXT,
    direct_flights_only BOOLEAN DEFAULT FALSE,
    max_layover_time INTEGER,
    preferred_departure_times JSONB,
    accommodation_types JSONB,
    hotel_rating INTEGER,
    amenities JSONB,
    neighborhood_preferences JSONB,
    transportation_modes JSONB,
    public_transport_preference BOOLEAN DEFAULT TRUE,
    car_rental_preference BOOLEAN DEFAULT FALSE,
    activity_types JSONB,
    pace_preference TEXT,
    cultural_interests JSONB,
    cuisine_preferences JSONB,
    special_interests JSONB,
    accessibility_requirements JSONB,
    dietary_restrictions JSONB,
    budget_allocation JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_travel_query_id ON user_preferences(travel_query_id);

-- Travel plans table
CREATE TABLE IF NOT EXISTS travel_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    travel_query_id UUID REFERENCES travel_queries(id),
    destination JSONB,
    flights JSONB,
    accommodation JSONB,
    transportation JSONB,
    activities JSONB,
    budget JSONB,
    overview TEXT,
    recommendations JSONB,
    alerts JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_travel_plans_user_id ON travel_plans(user_id);
CREATE INDEX IF NOT EXISTS idx_travel_plans_travel_query_id ON travel_plans(travel_query_id);
CREATE INDEX IF NOT EXISTS idx_travel_plans_created_at ON travel_plans(created_at);

-- Add a GIN index for JSONB destination field to support efficient text search
CREATE INDEX IF NOT EXISTS idx_travel_plans_destination_gin ON travel_plans USING GIN (destination jsonb_path_ops);

-- Create a view for travel plan summaries
CREATE OR REPLACE VIEW travel_plan_summaries AS
SELECT 
    tp.id,
    tp.user_id,
    tp.travel_query_id,
    tp.destination->>'name' as destination_name,
    tp.destination->>'country' as destination_country,
    tq.departure_date,
    tq.return_date,
    tq.travelers,
    tp.budget->>'total_cost' as total_cost,
    tp.created_at,
    tp.updated_at
FROM 
    travel_plans tp
JOIN 
    travel_queries tq ON tp.travel_query_id = tq.id;

-- Add a function to search travel plans by text
CREATE OR REPLACE FUNCTION search_travel_plans(search_text TEXT)
RETURNS SETOF travel_plans AS $$
BEGIN
    RETURN QUERY 
    SELECT * 
    FROM travel_plans
    WHERE 
        destination->>'name' ILIKE '%' || search_text || '%' OR
        destination->>'country' ILIKE '%' || search_text || '%' OR
        overview ILIKE '%' || search_text || '%';
END;
$$ LANGUAGE plpgsql;