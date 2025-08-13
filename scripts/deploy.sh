#!/bin/bash

# HFT Simulator Cloud-Native Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE=${NAMESPACE:-"hft-system"}
ENVIRONMENT=${ENVIRONMENT:-"development"}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"hft-simulator"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists kubectl; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    if ! command_exists docker; then
        print_error "docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command_exists helm; then
        print_warning "helm is not installed. Some features may not be available."
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info >/dev/null 2>&1; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    print_success "Prerequisites check completed"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    local services=("orderbook" "strategy" "marketdata" "dashboard")
    
    for service in "${services[@]}"; do
        print_status "Building $service service..."
        docker build -f "docker/services/Dockerfile.$service" -t "$DOCKER_REGISTRY/$service:$IMAGE_TAG" .
        
        if [ $? -eq 0 ]; then
            print_success "$service image built successfully"
        else
            print_error "Failed to build $service image"
            exit 1
        fi
    done
    
    print_success "All Docker images built successfully"
}

# Push images to registry (if not local registry)
push_images() {
    if [[ "$DOCKER_REGISTRY" != *"localhost"* ]] && [[ "$DOCKER_REGISTRY" != *"127.0.0.1"* ]]; then
        print_status "Pushing Docker images to registry..."
        
        local services=("orderbook" "strategy" "marketdata" "dashboard")
        
        for service in "${services[@]}"; do
            print_status "Pushing $service image..."
            docker push "$DOCKER_REGISTRY/$service:$IMAGE_TAG"
            
            if [ $? -eq 0 ]; then
                print_success "$service image pushed successfully"
            else
                print_error "Failed to push $service image"
                exit 1
            fi
        done
    else
        print_status "Using local registry, skipping image push"
    fi
}

# Create namespace
create_namespace() {
    print_status "Creating namespace $NAMESPACE..."
    
    if kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        print_warning "Namespace $NAMESPACE already exists"
    else
        kubectl apply -f k8s/base/namespace.yaml
        print_success "Namespace $NAMESPACE created"
    fi
}

# Deploy infrastructure components
deploy_infrastructure() {
    print_status "Deploying infrastructure components..."
    
    # Deploy Redis
    print_status "Deploying Redis..."
    kubectl apply -f k8s/base/redis.yaml
    
    # Deploy Kafka
    print_status "Deploying Kafka..."
    kubectl apply -f k8s/base/kafka.yaml
    
    # Wait for infrastructure to be ready
    print_status "Waiting for infrastructure components to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=kafka -n "$NAMESPACE" --timeout=300s
    
    print_success "Infrastructure components deployed successfully"
}

# Deploy application services
deploy_services() {
    print_status "Deploying application services..."
    
    # Update image tags in manifests
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|image: hft-simulator/|image: $DOCKER_REGISTRY/|g" k8s/base/services.yaml
        sed -i '' "s|:latest|:$IMAGE_TAG|g" k8s/base/services.yaml
    else
        # Linux
        sed -i "s|image: hft-simulator/|image: $DOCKER_REGISTRY/|g" k8s/base/services.yaml
        sed -i "s|:latest|:$IMAGE_TAG|g" k8s/base/services.yaml
    fi
    
    kubectl apply -f k8s/base/services.yaml
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    kubectl wait --for=condition=ready pod -l component=core -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l component=ui -n "$NAMESPACE" --timeout=300s
    
    print_success "Application services deployed successfully"
}

# Deploy Istio service mesh (if available)
deploy_istio() {
    if command_exists istioctl; then
        print_status "Deploying Istio service mesh configuration..."
        kubectl apply -f k8s/base/istio.yaml
        print_success "Istio configuration deployed"
    else
        print_warning "istioctl not found. Skipping Istio deployment."
    fi
}

# Deploy monitoring and autoscaling
deploy_monitoring() {
    print_status "Deploying monitoring and autoscaling..."
    kubectl apply -f k8s/base/autoscaling.yaml
    print_success "Monitoring and autoscaling deployed"
}

# Verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Check pod status
    print_status "Pod status:"
    kubectl get pods -n "$NAMESPACE"
    
    # Check service status
    print_status "Service status:"
    kubectl get services -n "$NAMESPACE"
    
    # Check ingress
    print_status "Ingress status:"
    kubectl get ingress -n "$NAMESPACE"
    
    # Wait for all pods to be ready
    local ready_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers | grep -c "1/1.*Running")
    local total_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l)
    
    if [ "$ready_pods" -eq "$total_pods" ]; then
        print_success "All pods are ready!"
    else
        print_warning "$ready_pods out of $total_pods pods are ready"
    fi
    
    # Get external IP
    local external_ip=$(kubectl get service dashboard-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    if [[ "$external_ip" != "pending" ]] && [[ "$external_ip" != "" ]]; then
        print_success "HFT Dashboard is accessible at: http://$external_ip"
    else
        print_warning "External IP is still pending. You can access the dashboard via port-forward:"
        echo "kubectl port-forward -n $NAMESPACE service/dashboard-service 8080:80"
    fi
}

# Port forward for local access
setup_port_forward() {
    print_status "Setting up port forwarding for local access..."
    
    echo "#!/bin/bash" > port-forward.sh
    echo "echo 'Setting up port forwarding...'" >> port-forward.sh
    echo "kubectl port-forward -n $NAMESPACE service/dashboard-service 8080:80 &" >> port-forward.sh
    echo "kubectl port-forward -n $NAMESPACE service/orderbook-service 8000:8000 &" >> port-forward.sh
    echo "kubectl port-forward -n $NAMESPACE service/strategy-service 8001:8001 &" >> port-forward.sh
    echo "kubectl port-forward -n $NAMESPACE service/marketdata-service 8002:8002 &" >> port-forward.sh
    echo "kubectl port-forward -n $NAMESPACE service/redis-master 6379:6379 &" >> port-forward.sh
    echo "kubectl port-forward -n $NAMESPACE service/kafka-service 9092:9092 &" >> port-forward.sh
    echo "echo 'Port forwarding setup complete. Access the dashboard at http://localhost:8080'" >> port-forward.sh
    echo "echo 'Press Ctrl+C to stop all port forwards'" >> port-forward.sh
    echo "wait" >> port-forward.sh
    
    chmod +x port-forward.sh
    print_success "Port forward script created: ./port-forward.sh"
}

# Main deployment function
main() {
    echo "=================================================="
    echo "HFT Simulator Cloud-Native Deployment"
    echo "=================================================="
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Docker Registry: $DOCKER_REGISTRY"
    echo "Image Tag: $IMAGE_TAG"
    echo "=================================================="
    
    check_prerequisites
    
    # Parse command line arguments
    case "${1:-deploy}" in
        "build")
            build_images
            ;;
        "push")
            push_images
            ;;
        "deploy")
            build_images
            push_images
            create_namespace
            deploy_infrastructure
            deploy_services
            deploy_istio
            deploy_monitoring
            verify_deployment
            setup_port_forward
            ;;
        "verify")
            verify_deployment
            ;;
        "clean")
            print_status "Cleaning up deployment..."
            kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
            docker system prune -f
            print_success "Cleanup completed"
            ;;
        "port-forward")
            setup_port_forward
            ./port-forward.sh
            ;;
        *)
            echo "Usage: $0 {build|push|deploy|verify|clean|port-forward}"
            echo ""
            echo "Commands:"
            echo "  build        - Build Docker images"
            echo "  push         - Push Docker images to registry"
            echo "  deploy       - Full deployment (build, push, deploy)"
            echo "  verify       - Verify deployment status"
            echo "  clean        - Clean up deployment"
            echo "  port-forward - Setup port forwarding for local access"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
