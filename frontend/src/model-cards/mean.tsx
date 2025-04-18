import { useEffect, useState } from "react";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import backendClient from "@/backendClient";
import Recommendation from "@/components/recommendation";
import { Restaurant } from "@/models/restaurant";


const Mean = (props: any) => {

    const [isLoading, setLoading] = useState(false);
    const [recommendations, setRecommendations] = useState<Restaurant[]>([]);
    const [userName, setUserName] = useState<string | null>(null);
    const [searched, setSearched] = useState(false);
    const { userId } = props;

    useEffect(() => {
        const handleUserFetch = async() => {
            if (userId == null) {
                return;
            }
            setLoading(true);
            setSearched(true);
            setRecommendations([]);
            setUserName(null);
            try {
                const result = await backendClient.get("/mean", {
                    params: {
                        user_id: userId
                    }
                });
                if (result.data && typeof result.data === 'object') {
                    setUserName(result.data.user_name || userId);
                    setRecommendations(Array.isArray(result.data.recommendations) ? result.data.recommendations : []);
                } else {
                    setUserName(userId);
                    setRecommendations([]);
                    console.error("Unexpected API response format:", result.data);
                }
            } catch (error) {
                console.error("Error fetching recommendations:", error);
                setRecommendations([]);
                setUserName(userId);
            } finally {
                setLoading(false);
            }
        }
        handleUserFetch();
    }, [userId]);


    console.log("in mean : ", userId)
    if(userId === null)
    {
        return <div>please select a persona from the left hand bar.</div>
    }

    return (
        <>
            <p className="mb-4 text-muted-foreground">
                Finds the highest-rated restaurants in the cities the user has previously reviewed.
            </p>

            {isLoading && <BackdropWithSpinner />} 

            {!isLoading && searched && recommendations.length === 0 && (
                <p>No recommendations found for this user, or the user has no reviews.</p>
            )}

            {recommendations.length > 0 && userName && (
                <div className="mt-6">
                    <h2 className="text-xl font-semibold mb-4">Recommendations for {userName}:</h2>
                    <div className="space-y-4">
                        {recommendations.map((recommendation) => (
                            <Recommendation key={recommendation.business_id} restaurant={recommendation} />
                        ))}
                    </div>
                </div>
            )}
        </>
    )
};


export default Mean;