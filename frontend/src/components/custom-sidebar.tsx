import { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"


export default function Sidebar(props:any) {
  const [collapsed, setCollapsed] = useState(false);

  const {children} = props;
  return (
    <div className="flex h-screen">
      <div
        className={cn(
          "transition-all duration-300 bg-gray-800 text-white p-4",
          collapsed ? "w-16" : "w-64"
        )}
      >
        <div className="flex justify-between items-center mb-4">
          {!collapsed && <h2 className="text-lg font-bold">Choose a persona!</h2>}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setCollapsed(!collapsed)}
            className="text-white hover:bg-gray-700"
          >
            {collapsed ? <ChevronRight /> : <ChevronLeft />}
          </Button>
        </div>

        <div className="space-y-4">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="bg-white text-black flex justify-evenly align-center rounded-3xl py-3"
              style={{ opacity: 0.8 }}
            >
              <Avatar className="my-auto">
                <AvatarImage src="https://github.com/shadcn.png" />
                <AvatarFallback>CN</AvatarFallback>
              </Avatar>
              {!collapsed && <div className="my-auto">Username</div>}
            </div>
          ))}
        </div>
      </div>

      <div className="flex-1 bg-gray-100 p-6">
        <h1 className="text-2xl font-semibold">{children}</h1>
      </div>
    </div>
  );
}