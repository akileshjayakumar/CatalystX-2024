# catalystX

## Project Structure

- **client/**: React frontend (Next.js).
  - **app/**: Routing and rendering.
  - **components/**: UI components.
  - **hooks/**: Custom hooks.
  - **lib/**: Utilities.
  - **build/**: Docker setup.
- **server/**: Python backend (FastAPI).
  - **src/**: API and AI logic.
  - **build/**: Docker setup.
- **docker-compose.yml**: Manages client and server with Docker.

## Prerequisites

Install:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Node.js](https://nodejs.org/) (for client)
- [Python 3.10+](https://www.python.org/) (for server)

## Environment Setup

1. **Clone the repo**:

   ```bash
   git clone https://github.com/akileshjayakumar/catalystX.git
   cd catalystX
   ```

2. **Set up environment variables**:

   ```bash
   cd server
   ```

   ```bash
   touch .env
   ```

   ```env
   OPENAI_API_KEY=your-openai-api-key
   ```

## Docker Setup

1. **Navigate to the project directory**:

   ```bash
   cd catalystX
   ```

2. **Build and run the Docker containers**:

   ```bash
   docker-compose up --build
   ```

3. **Access the services**:

   - Frontend: [http://localhost:3000](http://localhost:3000)
   - Backend: [http://localhost:8000](http://localhost:8000)
