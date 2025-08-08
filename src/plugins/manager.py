from typing import Type, Any, Dict, List
from src.plugins.base import DBConnector

class PluginManager:
    """
    A manager class for registering and retrieving database connector plugins.
    Implements the singleton pattern to ensure consistent plugin registration.
    """
    
    _instance = None
    _plugins: Dict[str, Type[DBConnector]] = {}
    _active_connections: Dict[str, DBConnector] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_plugin(cls, name: str, plugin_cls: Type[DBConnector]) -> None:
        """
        Register a new database connector plugin.
        
        Args:
            name (str): The name under which to register the plugin
            plugin_cls (Type[DBConnector]): The plugin class to register
            
        Raises:
            ValueError: If plugin name is already registered
        """
        if name in cls._plugins:
            raise ValueError(f"Plugin already registered under name: {name}")
        if not issubclass(plugin_cls, DBConnector):
            raise ValueError(f"Plugin class must inherit from DBConnector")
        cls._plugins[name] = plugin_cls

    @classmethod
    def get_plugin(cls, name: str, **kwargs: Any) -> DBConnector:
        """
        Retrieve and instantiate a registered plugin.
        
        Args:
            name (str): The name of the plugin to retrieve
            **kwargs: Additional arguments to pass to the plugin constructor
            
        Returns:
            DBConnector: An instance of the requested plugin
            
        Raises:
            ValueError: If no plugin is registered under the given name
        """
        plugin_cls = cls._plugins.get(name)
        if plugin_cls is None:
            raise ValueError(f"No plugin registered under the name: {name}")
            
        # Create new instance or return existing one
        if name not in cls._active_connections:
            cls._active_connections[name] = plugin_cls(**kwargs)
        return cls._active_connections[name]

    @classmethod
    def list_plugins(cls) -> List[str]:
        """List all registered plugin names"""
        return list(cls._plugins.keys())

    @classmethod
    def cleanup(cls) -> None:
        """Clean up all active connections"""
        for connection in cls._active_connections.values():
            try:
                connection.disconnect()
            except Exception:
                pass
        cls._active_connections.clear()

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()