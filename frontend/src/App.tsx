import './App.css'
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card } from "@/components/ui/card"
import DeepLearning from './model-cards/deeplearning'
import Traditional from './model-cards/traditional'
import Mean from './model-cards/mean'
import Sidebar from './components/custom-sidebar'
import { useState } from 'react'

function App() {

  const [userId, setUserId] = useState(null);
  const [userName, setUserName] = useState(null);
  console.log("User Id set : ", userId);

  return (
    <>
    <div>
      <Sidebar
        setUserId={setUserId}
        setUserName={setUserName}>
      <div className="header p-6 text-xl border-b">Restaurant Recommender</div>
    <div className="min-h-screen p-8 pb-8 sm:p-8">      
      <main className="max-w-4xl mx-auto flex flex-col gap-16">
      <div>
      <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl">
        Find Your Next Favorite Restaurant!
      </h1>
      <p className="leading-7 [&:not(:first-child)]:mt-6 m-6 sm:m-6">
        Explore different recommendation approaches to discover restaurants based on a user's preferences and history.
      </p>
      <Tabs defaultValue="naive">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="naive">Naive model</TabsTrigger>
          <TabsTrigger value="traditional">Traditional Model</TabsTrigger>
          <TabsTrigger value="deep-learning">Deep Learning Model</TabsTrigger>
        </TabsList>
        <TabsContent value="naive">
            <Mean 
              userId={userId}
            />
        </TabsContent>
        <TabsContent value="traditional">
            <Traditional 
              userId={userId}
              userName={userName} />
        </TabsContent>
        <TabsContent value="deep-learning">
            <DeepLearning 
              userId={userId}
              userName={userName} />
        </TabsContent>
      </Tabs>
      </div>


      </main>

    </div>
    </Sidebar>
    </div>

      
    </>
  )
}

export default App
