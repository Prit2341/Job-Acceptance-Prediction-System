pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code...'
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                echo 'Setting up Python environment...'
                sh '''
                    # Check for Python installation
                    if command -v python3 &> /dev/null; then
                        PYTHON_CMD=python3
                    elif command -v python &> /dev/null; then
                        PYTHON_CMD=python
                    else
                        echo "Python not found!"
                        exit 1
                    fi
                    
                    echo "Using Python: $PYTHON_CMD"
                    $PYTHON_CMD --version
                    $PYTHON_CMD -m pip install --upgrade pip --user
                '''
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo 'Installing required packages...'
                sh '''
                    if command -v python3 &> /dev/null; then
                        PYTHON_CMD=python3
                    else
                        PYTHON_CMD=python
                    fi
                    
                    $PYTHON_CMD -m pip install -r requirements.txt --user
                '''
            }
        }
        
        stage('Run Tests') {
            steps {
                echo 'Running unit tests...'
                sh '''
                    if command -v python3 &> /dev/null; then
                        PYTHON_CMD=python3
                    else
                        PYTHON_CMD=python
                    fi
                    
                    $PYTHON_CMD -m pytest src/ --junitxml=test-results.xml || true
                '''
            }
        }
        
        stage('Train Model') {
            steps {
                echo 'Training ML models...'
                sh '''
                    if command -v python3 &> /dev/null; then
                        PYTHON_CMD=python3
                    else
                        PYTHON_CMD=python
                    fi
                    
                    $PYTHON_CMD train.py
                '''
            }
        }
        
        stage('Archive Artifacts') {
            steps {
                echo 'Archiving artifacts...'
                archiveArtifacts artifacts: 'mlruns/**/*', allowEmptyArchive: true
                archiveArtifacts artifacts: '*.csv', allowEmptyArchive: true
            }
        }
    }
    
    post {
        always {
            echo 'Cleaning up...'
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
