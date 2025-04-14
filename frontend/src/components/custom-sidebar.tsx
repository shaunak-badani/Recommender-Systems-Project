import { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Card } from "./ui/card";


export default function Sidebar(props:any) {
  const [collapsed, setCollapsed] = useState(false);

  const preLoadedUsers = [
    {
      "user_id": "oc5N8oq1SY6quhq7U2Ep3A",
      "name": "Kim"
    },
    {
      "user_id": "3PTtt4JZD16P-TtC8HAbzg",
      "name": "Lisa"
    },
    {
      "user_id": "TngwxDH7eP2514kjevae_Q",
      "name": "Telly"
    },
    {
      "user_id": "RbNQqE8OwWOzF4qbGySxDQ",
      "name": "Amirah"
    },
    {
      "user_id": "d4PyzKtvlAyK7ZI67Mr2xA",
      "name": "Jieqi"
    },
    {
      "user_id": "9lxeey8azLxu_DhcsrCO2Q",
      "name": "David"
    },
    {
      "user_id": "lkCrRGX57VO9sfhIk1KVDw",
      "name": "Hannah"
    },
    {
      "user_id": "kk4GynEiF13My49uJ3hP7w",
      "name": "Joe"
    },
    {
      "user_id": "3nPXXCuPD64fW506XdRjGg",
      "name": "Ed"
    },
    {
      "user_id": "6AKD3ZTgkBQjGMZKhzkuDg",
      "name": "Jason"
    }
  ]

  const {children, setUserId, setUserName} = props;


  const activateUser = (userIndex: number) => {
    let user = preLoadedUsers[userIndex];
    setUserId(user.user_id);
    setUserName(user.name);
  }

  return (
    <div className="flex h-screen">
      <div
        className={cn(
          "transition-all duration-300 p-4 border-r-4",
          // collapsed ? "w-24" : "w-64"
        )}
      >
        <div className="flex justify-between items-center mb-4">
          {!collapsed && <h2 className="text-lg font-bold">Choose a persona!</h2>}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setCollapsed(!collapsed)}
            className="hover:bg-gray-700"
          >
            {collapsed ? <ChevronRight /> : <ChevronLeft />}
          </Button>
        </div>

        <div className="space-y-4">
          {preLoadedUsers.map((user, index) => (
              <Card
                onClick={() => activateUser(index)}
                key={user.user_id}
                className="flex justify-evenly align-center rounded-3xl py-3"
                style={{ opacity: 0.8 }}
              >
                <Avatar className="my-auto">
                  <AvatarImage src="https://github.com/shadcn.png" />
                  <AvatarFallback>CN</AvatarFallback>
                </Avatar>
                {!collapsed && <div className="my-auto">{user.name}</div>}
              </Card>
          ))}
        </div>
      </div>

      <div className="flex-1 p-6">
        <h1 className="text-2xl font-semibold">{children}</h1>
      </div>
    </div>
  );
}