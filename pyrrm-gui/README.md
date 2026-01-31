# pyrrm-gui

A web-based graphical user interface for the pyrrm (Python Rainfall-Runoff Models) library. This application allows hydrologists to configure, run, and analyze model calibration experiments through an intuitive web interface.

## Features

- **Catchment Management**: Create and manage hydrological catchments with their associated data
- **Dataset Upload**: Upload rainfall, PET, and observed flow data in CSV format
- **Experiment Configuration**: Step-by-step wizard for configuring calibration experiments
- **Asynchronous Calibration**: Run calibrations in the background while continuing to work
- **Real-time Progress**: Monitor calibration progress via WebSocket updates
- **Interactive Results**: View hydrographs, flow duration curves, and performance metrics
- **Experiment Comparison**: Compare multiple calibration results side-by-side

## Supported Models

- **Sacramento** - 20-parameter conceptual model
- **GR4J** - 4-parameter daily model
- **GR5J** - 5-parameter extension of GR4J
- **GR6J** - 6-parameter extension with exponential store

## Calibration Algorithms

- **SCE-UA Direct** - Shuffled Complex Evolution (recommended)
- **Differential Evolution** - SciPy global optimization
- **DREAM** - Bayesian MCMC (requires SpotPy)
- **PyDREAM** - MT-DREAM(ZS) (requires PyDREAM)

## Quick Start

### Using Docker (Recommended)

1. **Prerequisites**
   - Docker and Docker Compose installed
   - pyrrm library in the parent directory

2. **Start the application**
   ```bash
   cd pyrrm-gui
   docker-compose up --build
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/api/docs

### Manual Setup

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

#### Redis (for background tasks)

```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or install Redis locally
```

#### Celery Worker (for background calibrations)

```bash
cd backend
celery -A celery_worker worker --loglevel=info
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Container                       │
│                                                           │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │   Frontend  │────│   Backend    │────│   Celery   │  │
│  │   (React)   │    │  (FastAPI)   │    │  Workers   │  │
│  └─────────────┘    └──────────────┘    └────────────┘  │
│         │                  │                   │          │
│         │                  │                   │          │
│         │           ┌──────┴──────┐           │          │
│         │           │   SQLite    │           │          │
│         │           │   Redis     │───────────┘          │
│         │           └─────────────┘                      │
│         │                  │                              │
│         └──────────────────┼──────────────────────────── │
│                            │                              │
│                     ┌──────┴──────┐                      │
│                     │    pyrrm    │                      │
│                     │   Library   │                      │
│                     └─────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

## API Documentation

Once the backend is running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## Workflow

1. **Create a Catchment**: Add basic catchment information (name, gauge ID, area)
2. **Upload Datasets**: Upload rainfall, PET, and observed flow CSV files
3. **Create Experiment**: Use the wizard to configure:
   - Select model type
   - Set parameter bounds
   - Choose objective function
   - Configure algorithm settings
4. **Run Calibration**: Start the calibration and monitor progress
5. **View Results**: Analyze results with interactive charts and metrics
6. **Compare**: Compare multiple experiments to find the best configuration

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | sqlite:///./data/pyrrm.db | Database connection string |
| REDIS_URL | redis://localhost:6379/0 | Redis connection string |
| DEBUG | false | Enable debug mode |
| DEFAULT_WARMUP_DAYS | 365 | Default warmup period |
| DEFAULT_MAX_EVALS | 50000 | Default max function evaluations |

### Data Directory Structure

```
data/
├── uploads/       # Uploaded CSV files
│   └── {catchment_id}/
├── reports/       # CalibrationReport files (.pkl)
└── checkpoints/   # Calibration checkpoints
    └── {experiment_id}/
```

## Development

### Project Structure

```
pyrrm-gui/
├── backend/
│   ├── app/
│   │   ├── api/           # API endpoints
│   │   ├── models/        # SQLAlchemy models
│   │   ├── schemas/       # Pydantic schemas
│   │   ├── services/      # Business logic
│   │   ├── tasks/         # Celery tasks
│   │   └── websocket/     # WebSocket handlers
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   ├── hooks/         # Custom hooks
│   │   ├── services/      # API client
│   │   └── types/         # TypeScript types
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

## Support

For questions or issues, please open a GitHub issue or contact the development team.
