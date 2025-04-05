import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import backendClient from "@/backendClient";


const Traditional = () => {

    const [isLoading, setLoading] = useState(false);
    const [query, setQuery] = useState("");
    const [response, setResponse] = useState("");

    const handlePromptInput = async(query: string) => {
        setLoading(true);
        const response = await backendClient.get("/traditional", {
            params: {
                query: query
            }
        });
        setResponse(response.data.answer);
        setLoading(false);
    }

    return (
        <>
            <h6 className="pb-6 sm:pb-6">Traditional model</h6>
            <p>Traditional model description here</p>
            <Textarea
                value={query}
                onChange = {(e) => setQuery(e.target.value)} 
                placeholder="Enter your query here!" />
            <Button className="p-6 sm:p-6 rounded-2xl m-8 sm:m-8" onClick={() => handlePromptInput(query)}>
                Send
            </Button>
            {response.length > 0 && <p>{response}</p>}
            {isLoading && <BackdropWithSpinner />}
        </>
    )
};


export default Traditional;