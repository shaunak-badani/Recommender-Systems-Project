# Template project

> A template project with React & Vite as frontend and fastapi as backend.


### How to run

- Frontend
```
cd frontend
npm install
npm run dev
```
- The application is then hosted on `localhost:5173`


- Backend
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
fastapi dev main.py
```


### Deploym

#### VCM

If you're deploying on vcm, change the vcm base url in `frontend/.env.production`. You can change the link to the url / ip address of the server you are hosting it on, if using GCP or Azure for deployment.

Commands to deploy:

```bash
cd TemplateProject # You can rename this, just make sure the current directory has the docker compose file
sudo docker compose -f docker-compose.yml up --build -d
```

#### Azure (this is a hodge podge of steps, to be cleaned)


Open cloud shell

1. Output resource groups:
```
az group list --output table
```
2. Use any of the resource group mentioned above. Create an App Service Plan. Then check that it has been created:
```
export RESOURCE_GROUP="DefaultResourceGroup-EUS2" # Change this to resource group available you have
export APP_SERVICE_PLAN="devilsplan"
az appservice plan create --name $APP_SERVICE_PLAN --resource-group $RESOURCE_GROUP --sku F1 --is-linux
az appservice plan list --output table
```
3. Upload the docker compose file to azure. This can be done in the cloudshell tab, using "Manage files" > "Download"
4. Create an Azure Web App for Containers:
```
export WEBAPP_NAME="devils-codex"
az webapp create --resource-group $RESOURCE_GROUP \
  --plan $APP_SERVICE_PLAN \
  --name $WEBAPP_NAME \
  --multicontainer-config-type compose \
  --multicontainer-config-file docker-compose-azure.yml
```

5. Create ACR 
```
export ACR_NAME="codexregistry"
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic 
```

6. Login into ACR, and get the access token on the cloud shell
```
az acr login --name $ACR_NAME --expose-token
```

7. 

```
az account list --output table # GET subscription id
export SUBSCRIPTION_ID=<subscription-id-from-above>
export ACCESS_TOKEN=<access-token-received-above>
export USERNAME=$(az account show --query user.name -o tsv)
```

Now, on your local machine, build your images

Check if your account has ACRPull or ACRPush permissions:
az role assignment list --assignee $(az account show --query user.name -o tsv) --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ContainerRegistry/registries/$ACR_NAME --output table

If you don't see ACRPull or ACRPush, assign the necessary role:
az role assignment create --assignee $(az account show --query user.name -o tsv) --role "ACRPush" --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ContainerRegistry/registries/$ACR_NAME

8. Enable admin credentials (on cloud shell) and push images to acr (on local machine)
```
az acr update -n $ACR_NAME --admin-enabled true
az acr credential show -n $ACR_NAME # COPY USERNAME AND PASSWORD TO BELOW COMMANDS
```

On your local machine, 
```
export ACR_NAME=<acr-name-from-above>
export USERNAME=<output from credential show command>
export PASSWORD=<output from credential show command>
docker login $ACR_NAME.azurecr.io -u $USERNAME -p $PASSWORD
# Now build the images and push to acr
sudo docker compose -f docker-compose.yml build
docker tag templateproject-frontend:latest codexregistry.azurecr.io/templateproject-frontend:latest
docker tag templateproject-backend:latest codexregistry.azurecr.io/templateproject-backend:latest
docker push $ACR_NAME.azurecr.io/templateproject-frontend:latest
docker push $ACR_NAME.azurecr.io/templateproject-backend:latest
```

On cloudshell, verify that the acr has the images:
```
az acr repository list -n $ACR_NAME --output table
```

9? Restart web app

az webapp config container set \
  --name $WEBAPP_NAME \
  --resource-group $RESOURCE_GROUP \
  --container-registry-url https://$ACR_NAME.azurecr.io \
  --container-registry-user $USERNAME \
  --container-registry-password $PASSWORD

az webapp restart --name $WEBAPP_NAME --resource-group $RESOURCE_GROUP

az webapp show --name $WEBAPP_NAME --resource-group $RESOURCE_GROUP --query "defaultHostName" -o tsv