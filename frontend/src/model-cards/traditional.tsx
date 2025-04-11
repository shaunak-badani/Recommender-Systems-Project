import { useEffect, useState } from "react";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import backendClient from "@/backendClient";
import Recommendation from "@/components/recommendation";

const Traditional = () => {

    const [isLoading, setLoading] = useState(false);
    const [response, setResponse] = useState([]);
    let userId = "qVc8ODYU5SZjKXVBgXdI7w" // Hardcoded, change later

    const getRecommendations = async() => {
        
        const response = await backendClient.get('/traditional', {
            params: {
                user_id: userId 
            }
        });
        setResponse(response.data);
    }

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            await getRecommendations();
            setLoading(false);
        };

        fetchData();
    }, []);

    return (
        <>
            <h1 className="scroll-m-20 text-2xl font-extrabold m-8 sm:m-8 tracking-tight lg:text-3xl">
                Top 10 recommendations for Walker
            </h1>
            {response.length > 0 && response.map(
                recommendation => <Recommendation
                    restaurant={recommendation} />
            )}
            {isLoading && <BackdropWithSpinner />}
        </>
    )
};


export default Traditional;