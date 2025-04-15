import { useEffect, useState } from "react";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import backendClient from "@/backendClient";
import Recommendation from "@/components/recommendation";

const Traditional = (props: any) => {

    const [isLoading, setLoading] = useState(false);
    const [response, setResponse] = useState([]);
    const { userId, userName } = props // Hardcoded, change later

    const getRecommendations = async() => {
        if(userId === null)
            return;
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
    }, [userId]);

    if(userId === null)
    {
        return(<p>Please select a persona from the left hand bar.</p>)
    }

    return (
        <>
            <h1 className="scroll-m-20 text-2xl font-extrabold m-8 sm:m-8 tracking-tight lg:text-3xl">
                Top 10 recommendations for {userName}
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