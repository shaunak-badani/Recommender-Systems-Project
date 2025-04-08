import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { MapPinned } from "lucide-react";


const Recommendation = (props: any) => {

    const { restaurant } = props;

    console.log("Restaurant : ", restaurant)

    const { name, address, image } = restaurant;

    return (
        <div className="flex justify-center m-6 sm:m-6">
          <Card className="w-3/4 flex">
            <img 
              src={`data:image/jpeg;base64,${image}`} 
              alt="Business"
              style={{ maxWidth: "40%", height: "auto" }} 
              />
            <div className="w-full flex flex-col justify-center">
              <CardHeader>
                <CardTitle>{name}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-row justify-evenly">
                    <div className="flex flex-row w-full items-center gap-4 justify-center">
                        <MapPinned />
                        <div className="flex flex-col space-y-1.5">
                            <div>{address}</div>
                        </div>
                    </div>
                </div>
              </CardContent>
            </div>
          </Card>
        </div>
      )

    return (
        
        <div>this is a recommendation!</div>
    )
};

export default Recommendation
