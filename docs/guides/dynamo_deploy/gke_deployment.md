# Deploying Dynamo on Google Kubernetes Engine

Link to YouTube video: https://youtu.be/Pym5VHIPWaI

## Prelimiaries 
gcloud projects create dynamo-gke --name="Dynamo Cloud"
gcloud config set project dynamo-gke

create billing account

gcloud services enable \
  container.googleapis.com \
  compute.googleapis.com \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com

# Create a dynamo cluster with 2 nodes
# We need at least e2-standard-4 because buildkit requires 3 cpu and 8GB mem at least.

gcloud container clusters create dynamo-cluster \
  --zone us-central1-a \
  --machine-type e2-standard-2 \
  --num-nodes 3 \
  --disk-size 20GB \
  --disk-type pd-standard \
  --image-type COS_CONTAINERD \
  --enable-network-policy \
  --no-enable-basic-auth \
  --no-enable-legacy-authorization \
  --enable-ip-alias \
  --network default \
  --subnetwork default

# delete
gcloud container clusters delete dynamo-cluster \
  --zone us-central1-a \
  --quiet

# New bigger cluster
gcloud container clusters create dynamo-cluster \
  --zone us-central1-a \
  --machine-type e2-standard-4 \
  --num-nodes 2 \
  --enable-autoscaling \
  --min-nodes 2 --max-nodes 4 \
  --disk-type pd-standard \
  --disk-size 50GB \
  --image-type COS_CONTAINERD \
  --enable-ip-alias \
  --enable-network-policy \
  --no-enable-basic-auth \
  --no-enable-legacy-authorization 


gcloud container clusters create dynamo-cluster \
  --project dynamo-gke \
  --zone $ZONE \
  --machine-type e2-standard-8 \
  --num-nodes 2 \
  --disk-type pd-balanced \
  --disk-size 200 \
  --enable-ip-alias \
  --enable-network-policy \
  --enable-autoupgrade --enable-autorepair

gcloud container clusters get-credentials dynamo-cluster --zone us-central1-a

gcloud compute firewall-rules create allow-dynamo \
  --allow tcp:80,tcp:443 \
  --target-tags gke-dynamo-cluster \
  --description "Allow traffic to Dynamo Cloud"

# List all node VMs that belong to the cluster
gcloud compute instances list --filter="name~'gke-dynamo-cluster'"

# Pick one instance and jump in via IAP
INSTANCE=$(gcloud compute instances list \
           --filter="name~'gke-dynamo-cluster' AND status=RUNNING" \
           --limit=1 --format='value(name)')
ZONE=us-central1-a        # change if you created the cluster elsewhere

gcloud compute ssh $INSTANCE --zone $ZONE --tunnel-through-iap


docker login docker.io                                   # enter Docker-Hub creds
export DOCKER_SERVER=docker.io/<your-username>
export IMAGE_TAG=latest

# Should I export docker credentials in kubernetes or local -> Local 
# INstall Earthly and Helm in Kubernetes or local -> Local 
# Is Dynamo Cloud in Kubernetes or local? -> Kube
# Local just need kubectl 

# In Google Cloud Consule, click Connect, get the kubectl credientials and config 
gcloud container clusters get-credentials dynamo-cluster --zone us-central1-a --project dynamo-gke

kubectl get nodes    


# On mac
brew install --cask docker
brew install earthly 
brew install helm

# At folder root (where the Earthfile is)
docker login                     
export DOCKER_SERVER=docker.io/faradawn
export IMAGE_TAG=latest
earthly --push +all-docker --DOCKER_SERVER=$DOCKER_SERVER --IMAGE_TAG=$IMAGE_TAG

# Check kubernetes 
kubectl get storageclass

# Install
export DOCKER_USERNAME=faradawn
export DOCKER_SERVER=docker.io/faradawn
export IMAGE_TAG=latest
export NAMESPACE=dynamo-cloud

cd deploy/cloud/helm
kubectl create namespace $NAMESPACE
kubectl config set-context --current --namespace=$NAMESPACE

# Need Ingress or Istio
kubectl create namespace ingress-nginx

helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --set controller.publishService.enabled=true

helm uninstall ingress-nginx -n ingress-nginx

# Note that Helm 3.18.0 had issue with Ngnix. Need to roll back to 3.17.3. Or wait for new patch

kubectl delete namespace dynamo-cloud

kubectl get pods -n dynamo-cloud

If builtkit pending, dynamo-cloud-dynamo-operator-buildkitd-0                          0/1     Pending            0                4h18m

It's because it needs  in buildkit.yaml
requests:
              cpu: 3
              memory: 8Gi


helm uninstall dynamo-cloud -n dynamo-cloud 

kubectl -n dynamo-cloud get pods | grep buildkitd


kubectl -n dynamo-cloud describe pod dynamo-cloud-dynamo-operator-buildkitd-0


helm upgrade -i dynamo-cloud platform/ \
  -f generated-values.yaml \
  -f my-overrides.yaml


Check PVC pending 