import { useEffect, useState } from "react";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import backendClient from "@/backendClient";
import Recommendation from "@/components/recommendation";
import { Restaurant } from "@/models/restaurant";

const DeepLearning = (props: any) => {
    const [isLoading, setLoading] = useState(false);
    const [recommendations, setRecommendations] = useState<Restaurant[]>([]);

    const {userId, userName} = props;

    const handlePromptInput = async() => {
        if (userId === null) {
            return;
        }

        setLoading(true);
        setRecommendations([]);
        try {
            const result = await backendClient.get("/deep-learning", {
                params: {
                    user_id: userId
                }
            });
            if (result.data && typeof result.data === 'object') {
                setRecommendations(Array.isArray(result.data.recommendations) ? result.data.recommendations : []);
            } else {
                setRecommendations([]);
                console.error("Unexpected API response format:", result.data);
            }
        } catch (error) {
            console.error("Error fetching recommendations:", error);
            setRecommendations([]);
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => {
        handlePromptInput()
    }, [userId]);

    if(userId === null)
    {
        return <div>please select a persona from the left hand bar.</div>;
    }

    return (
        <>
            {isLoading && <BackdropWithSpinner />} 

            {!isLoading && recommendations.length === 0 && (
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

export default DeepLearning;