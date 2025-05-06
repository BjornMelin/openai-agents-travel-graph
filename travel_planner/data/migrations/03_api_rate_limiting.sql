-- API Rate Limiting and Request Tracking

-- Create a table to track API requests
CREATE TABLE IF NOT EXISTS api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    request_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_time INTEGER, -- in milliseconds
    success BOOLEAN,
    error_message TEXT,
    user_id UUID REFERENCES users(id),
    travel_query_id UUID REFERENCES travel_queries(id),
    request_payload JSONB,
    response_payload JSONB,
    metadata JSONB
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_api_requests_service_name ON api_requests(service_name);
CREATE INDEX IF NOT EXISTS idx_api_requests_request_time ON api_requests(request_time);
CREATE INDEX IF NOT EXISTS idx_api_requests_user_id ON api_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_api_requests_travel_query_id ON api_requests(travel_query_id);

-- Create a table to track rate limits for each service
CREATE TABLE IF NOT EXISTS api_rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name TEXT UNIQUE,
    requests_per_minute INTEGER,
    requests_per_hour INTEGER,
    requests_per_day INTEGER,
    cooldown_period_ms INTEGER,
    retry_backoff_factor FLOAT,
    max_retries INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default rate limits for common services
INSERT INTO api_rate_limits (
    service_name, 
    requests_per_minute, 
    requests_per_hour, 
    requests_per_day, 
    cooldown_period_ms, 
    retry_backoff_factor, 
    max_retries
) VALUES
('openai', 60, 3500, 80000, 1000, 2.0, 5),
('tavily', 10, 500, 10000, 2000, 2.0, 3),
('firecrawl', 5, 250, 5000, 3000, 2.0, 3),
('skyscanner', 5, 200, 4000, 5000, 2.0, 3),
('booking', 3, 150, 3000, 5000, 2.0, 3),
('airbnb', 3, 150, 3000, 5000, 2.0, 3),
('hotels', 5, 200, 4000, 5000, 2.0, 3),
('google_maps', 10, 600, 12000, 2000, 2.0, 3),
('google_flights', 5, 200, 4000, 5000, 2.0, 3),
('tripadvisor', 5, 200, 4000, 5000, 2.0, 3)
ON CONFLICT (service_name) DO UPDATE SET
    requests_per_minute = EXCLUDED.requests_per_minute,
    requests_per_hour = EXCLUDED.requests_per_hour,
    requests_per_day = EXCLUDED.requests_per_day,
    cooldown_period_ms = EXCLUDED.cooldown_period_ms,
    retry_backoff_factor = EXCLUDED.retry_backoff_factor,
    max_retries = EXCLUDED.max_retries,
    updated_at = NOW();

-- Function to check if a request should be allowed based on rate limits
CREATE OR REPLACE FUNCTION check_rate_limit(p_service_name TEXT) 
RETURNS TABLE(
    allowed BOOLEAN, 
    cooldown_ms INTEGER, 
    reason TEXT
) AS $$
DECLARE
    limits RECORD;
    minute_count INTEGER;
    hour_count INTEGER;
    day_count INTEGER;
BEGIN
    -- Get the rate limits for the service
    SELECT * INTO limits FROM api_rate_limits 
    WHERE service_name = p_service_name;
    
    IF NOT FOUND THEN
        -- Service not found in rate limits, use default conservative limits
        RETURN QUERY SELECT 
            TRUE as allowed, 
            0 as cooldown_ms, 
            'Service not in rate limit table, using default allowance' as reason;
        RETURN;
    END IF;
    
    -- Count requests in the last minute
    SELECT COUNT(*) INTO minute_count FROM api_requests
    WHERE service_name = p_service_name
    AND request_time > NOW() - INTERVAL '1 minute';
    
    -- Check minute limit
    IF minute_count >= limits.requests_per_minute THEN
        RETURN QUERY SELECT 
            FALSE as allowed, 
            limits.cooldown_period_ms as cooldown_ms, 
            'Exceeded requests per minute limit' as reason;
        RETURN;
    END IF;
    
    -- Count requests in the last hour
    SELECT COUNT(*) INTO hour_count FROM api_requests
    WHERE service_name = p_service_name
    AND request_time > NOW() - INTERVAL '1 hour';
    
    -- Check hour limit
    IF hour_count >= limits.requests_per_hour THEN
        RETURN QUERY SELECT 
            FALSE as allowed, 
            limits.cooldown_period_ms * 5 as cooldown_ms, 
            'Exceeded requests per hour limit' as reason;
        RETURN;
    END IF;
    
    -- Count requests in the last day
    SELECT COUNT(*) INTO day_count FROM api_requests
    WHERE service_name = p_service_name
    AND request_time > NOW() - INTERVAL '1 day';
    
    -- Check day limit
    IF day_count >= limits.requests_per_day THEN
        RETURN QUERY SELECT 
            FALSE as allowed, 
            limits.cooldown_period_ms * 20 as cooldown_ms, 
            'Exceeded requests per day limit' as reason;
        RETURN;
    END IF;
    
    -- All checks passed
    RETURN QUERY SELECT 
        TRUE as allowed, 
        0 as cooldown_ms, 
        'Request allowed' as reason;
END;
$$ LANGUAGE plpgsql;

-- Create a view to monitor API usage
CREATE OR REPLACE VIEW api_usage_stats AS
SELECT
    service_name,
    COUNT(*) AS total_requests,
    COUNT(*) FILTER (WHERE success = TRUE) AS successful_requests,
    COUNT(*) FILTER (WHERE success = FALSE) AS failed_requests,
    ROUND(AVG(response_time)) AS avg_response_time_ms,
    MAX(response_time) AS max_response_time_ms,
    MIN(request_time) AS first_request,
    MAX(request_time) AS last_request,
    -- Requests in different time periods
    COUNT(*) FILTER (WHERE request_time > NOW() - INTERVAL '5 minutes') AS requests_5min,
    COUNT(*) FILTER (WHERE request_time > NOW() - INTERVAL '1 hour') AS requests_1hour,
    COUNT(*) FILTER (WHERE request_time > NOW() - INTERVAL '24 hours') AS requests_24hours,
    -- Error rate
    ROUND(
        (COUNT(*) FILTER (WHERE success = FALSE))::DECIMAL / 
        NULLIF(COUNT(*), 0) * 100, 
        2
    ) AS error_rate_percent
FROM
    api_requests
GROUP BY
    service_name
ORDER BY
    total_requests DESC;