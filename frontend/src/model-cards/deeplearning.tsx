import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import backendClient from "@/backendClient";
import Recommendation from "@/components/recommendation";
import { Restaurant } from "@/models/restaurant";

const DeepLearning = () => {
    const [isLoading, setLoading] = useState(false);
    const [userId, setUserId] = useState("");
    const [recommendations, setRecommendations] = useState<Restaurant[]>([]);
    const [userName, setUserName] = useState<string | null>(null);
    const [searched, setSearched] = useState(false);

    const handlePromptInput = async() => {
        if (!userId.trim()) {
            return;
        }
        setLoading(true);
        setSearched(true);
        setRecommendations([]);
        setUserName(null);
        try {
            const result = await backendClient.get("/deep-learning", {
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

    return (
        <>
            <div className="flex w-full max-w-lg items-center space-x-2 mb-6">
                <Textarea
                    value={userId}
                    onChange = {(e) => setUserId(e.target.value)}
                    placeholder="Enter your User ID" 
                    rows={1}
                    className="min-h-[40px]"
                />
                <Button className="p-4" onClick={handlePromptInput} disabled={isLoading}>
                    {isLoading ? 'Searching...' : 'Find Restaurants'}
                </Button>
            </div>

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

export default DeepLearning;