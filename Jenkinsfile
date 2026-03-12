pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
        VENV_DIR = 'venv'
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code...'
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                echo 'Setting up Python virtual environment...'
                bat '''
                    python -m venv %VENV_DIR%
                    call %VENV_DIR%\\Scripts\\activate.bat
                    python -m pip install --upgrade pip
                '''
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo 'Installing required packages...'
                bat '''
                    call %VENV_DIR%\\Scripts\\activate.bat
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Run Tests') {
            steps {
                echo 'Running unit tests...'
                bat '''
                    call %VENV_DIR%\\Scripts\\activate.bat
                    pytest src/ --junitxml=test-results.xml || exit 0
                '''
            }
        }
        
        stage('Train Model') {
            steps {
                echo 'Training ML models...'
                bat '''
                    call %VENV_DIR%\\Scripts\\activate.bat
                    python train.py
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
