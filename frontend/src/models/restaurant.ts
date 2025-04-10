export interface Restaurant {
    business_id: string;
    name: string;
    address?: string | null;
    city?: string | null;
    state?: string | null;
    stars?: number | null;
    review_count?: number | null;
}