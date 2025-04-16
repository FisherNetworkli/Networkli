-- Function to get database size and statistics
CREATE OR REPLACE FUNCTION public.get_database_info()
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    result json;
BEGIN
    SELECT json_build_object(
        'total_size', pg_database_size(current_database()),
        'table_count', (SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'),
        'index_count', (SELECT count(*) FROM pg_indexes WHERE schemaname = 'public')
    ) INTO result;
    
    RETURN result;
END;
$$;

-- Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION public.get_database_info() TO authenticated;
GRANT EXECUTE ON FUNCTION public.get_database_info() TO service_role; 