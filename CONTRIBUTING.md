# Contributing to HFT Simulator 🤝

We welcome contributions to the HFT Simulator project! This document provides guidelines for contributing to the project.

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- Use GitHub Issues to report bugs
- Include detailed steps to reproduce
- Provide system information and error messages
- Check if the issue already exists

### ✨ Feature Requests
- Open an issue with the "enhancement" label
- Describe the feature and its use case
- Explain how it fits with the project goals
- Consider implementation complexity

### 📝 Documentation
- Improve existing documentation
- Add usage examples
- Create tutorials or guides
- Fix typos and clarifications

### 💻 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Test coverage improvements

## 🔧 Development Setup

### Prerequisites
- Python 3.11+
- Git
- Virtual environment tools

### Setup Steps
```bash
# Fork and clone the repository
git clone https://github.com/your-username/hft-simulator.git
cd hft-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run development setup
python dev_setup.py
```

### Development Environment
```bash
# Install development dependencies
pip install -r requirements-ml.txt  # For ML features
pip install -r requirements-realtime.txt  # For real-time features

# Run tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

## 📋 Contribution Process

### 1. Fork & Branch
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/your-username/hft-simulator.git

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. Development Guidelines

#### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular

#### Project Structure
- Production code goes in `src/`
- Tests go in `tests/` with appropriate categorization
- Documentation goes in `docs/`
- Examples go in `examples/`
- Configuration in `config/`

#### Testing Requirements
- Add unit tests for new functionality
- Include integration tests for system features
- Ensure all tests pass before submitting
- Maintain or improve test coverage

### 3. Commit Guidelines

#### Commit Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

#### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or fixing tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

#### Examples
```bash
feat(engine): add order book depth calculation
fix(strategies): resolve market making inventory issue
docs(README): update installation instructions
test(integration): add end-to-end trading tests
```

### 4. Pull Request Process

#### Before Submitting
- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with master

#### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally

## Documentation
- [ ] Code comments added
- [ ] Documentation updated
- [ ] Examples provided if needed
```

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Individual component tests
├── integration/    # System interaction tests
├── performance/    # Benchmarking and optimization tests
├── fixtures/       # Test data and utilities
└── conftest.py     # Shared test configuration
```

### Writing Tests
```python
import pytest
from src.engine.order_book import OrderBook

class TestOrderBook:
    def test_add_order(self):
        """Test basic order addition functionality."""
        book = OrderBook()
        order = {"id": 1, "side": "buy", "price": 100, "quantity": 10}
        book.add_order(order)
        assert len(book.bids) == 1
        
    def test_order_matching(self):
        """Test order matching logic."""
        # Test implementation
        pass
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/unit/test_order_book.py

# Run with verbose output
pytest -v tests/integration/
```

## 📊 Performance Considerations

### Optimization Guidelines
- Profile code before optimizing
- Use vectorized operations where possible
- Consider memory usage for large datasets
- Benchmark performance improvements

### Performance Tests
- Add benchmarks for critical paths
- Test with realistic data volumes
- Monitor memory usage
- Validate latency requirements

## 🔍 Code Review Process

### Review Checklist
- [ ] Code functionality works as intended
- [ ] Tests provide adequate coverage
- [ ] Documentation is clear and complete
- [ ] Performance impact is acceptable
- [ ] Security considerations addressed

### Reviewer Guidelines
- Be constructive and respectful
- Focus on code quality and project goals
- Suggest improvements where helpful
- Approve when requirements are met

## 📞 Getting Help

### Resources
- **Documentation**: Check `docs/` directory
- **Examples**: Review `examples/` for usage patterns
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions

### Contact
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For general questions
- **Pull Requests**: For code contributions

## 🎉 Recognition

Contributors are recognized in:
- GitHub contributor graphs
- Release notes for significant contributions
- Documentation acknowledgments

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to HFT Simulator! Your contributions help make quantitative finance education and research more accessible to everyone.** 🚀
