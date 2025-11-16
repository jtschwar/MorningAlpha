#!/usr/bin/env python3
"""
API key management for morningalpha.
Stores encrypted keys in the package installation directory.
"""
import json
import hashlib
import platform
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet

# Store keys in package directory
KEYS_FILE = Path(__file__).parent / ".api_keys.enc"
SALT_FILE = Path(__file__).parent / ".key_salt"


def _get_machine_key() -> bytes:
    """
    Generate a machine-specific encryption key.
    This uses hardware/system identifiers to create a unique key per machine.
    """
    # Combine multiple machine identifiers
    machine_id = platform.node()  # hostname
    machine_platform = platform.platform()  # OS info
    
    # Create a consistent machine fingerprint
    fingerprint = f"{machine_id}:{machine_platform}".encode()
    
    # Load or create salt
    if SALT_FILE.exists():
        with open(SALT_FILE, "rb") as f:
            salt = f.read()
    else:
        # Generate random salt on first run
        import secrets
        salt = secrets.token_bytes(32)
        SALT_FILE.write_bytes(salt)
        SALT_FILE.chmod(0o600)
    
    # Derive encryption key using PBKDF2
    key_material = hashlib.pbkdf2_hmac(
        'sha256',
        fingerprint,
        salt,
        iterations=100000,
        dklen=32
    )
    
    # Encode as base64 for Fernet
    from base64 import urlsafe_b64encode
    return urlsafe_b64encode(key_material)


def _get_cipher() -> Fernet:
    """Get the Fernet cipher for encryption/decryption."""
    key = _get_machine_key()
    return Fernet(key)


def _load_keys() -> dict:
    """Load and decrypt all API keys from storage."""
    if not KEYS_FILE.exists():
        return {}
    
    try:
        cipher = _get_cipher()
        encrypted_data = KEYS_FILE.read_bytes()
        decrypted_data = cipher.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode('utf-8'))
    except Exception:
        # If decryption fails, return empty dict
        # (file might be corrupted or from different machine)
        return {}


def _save_keys(keys: dict) -> None:
    """Encrypt and save all API keys to storage."""
    try:
        cipher = _get_cipher()
        json_data = json.dumps(keys, indent=2).encode('utf-8')
        encrypted_data = cipher.encrypt(json_data)
        
        KEYS_FILE.write_bytes(encrypted_data)
        KEYS_FILE.chmod(0o600)
    except Exception as e:
        raise RuntimeError(f"Failed to save API keys: {e}")


def get_key(service: str) -> Optional[str]:
    """
    Get an API key for a service.
    
    Args:
        service: Service name (e.g., 'alpha_vantage')
    
    Returns:
        API key string or None if not found
    """
    keys = _load_keys()
    return keys.get(service)


def set_key(service: str, api_key: str) -> None:
    """
    Store an encrypted API key for a service.
    
    Args:
        service: Service name (e.g., 'alpha_vantage')
        api_key: The API key to store (will be encrypted)
    """
    keys = _load_keys()
    keys[service] = api_key
    _save_keys(keys)


def delete_key(service: str) -> bool:
    """
    Delete an API key for a service.
    
    Args:
        service: Service name (e.g., 'alpha_vantage')
    
    Returns:
        True if key was deleted, False if it didn't exist
    """
    keys = _load_keys()
    if service in keys:
        del keys[service]
        _save_keys(keys)
        return True
    return False


def list_services() -> list[str]:
    """
    List all services with stored API keys.
    
    Returns:
        List of service names
    """
    keys = _load_keys()
    return list(keys.keys())


def clear_all_keys() -> None:
    """Delete all stored API keys and encryption metadata."""
    if KEYS_FILE.exists():
        KEYS_FILE.unlink()
    if SALT_FILE.exists():
        SALT_FILE.unlink()


# Convenience functions for Alpha Vantage
def get_alpha_vantage_key() -> Optional[str]:
    """Get the Alpha Vantage API key."""
    return get_key("alpha_vantage")


def set_alpha_vantage_key(api_key: str) -> None:
    """Set the Alpha Vantage API key."""
    set_key("alpha_vantage", api_key)


def has_alpha_vantage_key() -> bool:
    """Check if Alpha Vantage API key is configured."""
    return get_alpha_vantage_key() is not None