---
name: git-commit-formatter
description: Formats git commit messages according to Conventional Commits 
  specification. Use when user asks to commit changes or write a commit message.
---

# Git Commit Formatter

When writing a git commit message, follow the Conventional Commits specification.

## Format
## Allowed Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation only
- **style**: Formatting, no code change
- **refactor**: Code restructuring
- **test**: Adding tests
- **chore**: Maintenance tasks

## Rules
1. Type is mandatory and lowercase
2. Scope is optional, in parentheses
3. Description starts lowercase, no period at end
4. Description must be under 72 characters
5. Body explains "what" and "why", not "how"

## Examples
- `feat(auth): add OAuth2 login support`
- `fix: resolve null pointer in user service`
- `docs(readme): update installation instructions`

## Decision Tree
- If adding new functionality → use `feat`
- If fixing a bug → use `fix`
- If changing documentation only → use `docs`
- If changing code style without logic change → use `style`
