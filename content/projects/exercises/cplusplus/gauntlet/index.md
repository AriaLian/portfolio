+++
title = "Gauntlet"
description = ""
summary = ""
featuredImage = ""
tags = ["Game Engine", "ECS", "SDL"]
draft = false
categories = [""]
collections = ["C++"]
+++

## Overview
This is a top-down Rogue-like game developed in C++ using the SDL (Simple DirectMedia Layer) library. The codebase of this game is organized into two parts: the Engine code and the Game code. 
The Engine code follows an ECS (Entity Component System) architecture, which is completely isolated from the Game code, and can be used to create other games.

{{< github repo="arialian/Project_02_Gauntlet" showThumbnail=false >}}

{{< button href="https://github.com/arialian/Project_02_Gauntlet" target="_self" >}}
{{< icon "github" >}} View on Github
{{< /button >}}

## Gameplay
Currently, the gameplay of Gauntlet involves the following elements:
- Player Controls: The player's movement is controlled using keyboard inputs (WSAD), and using mouse click to fire a projectile towards the click point. 
- Enemy Spawn Factory: Enemy spawn factory will spawn enemies based on a timer. The player must avoid enemies that spawn into the room.
- Diverse Enemy Behaviors: The game features multiple enemies, each with its own behaviours.

## Engine Systems
The Engine incorporates the following systems, which are currently in development: 

### Asset Manager
The Asset Manager system is responsible for managing game assets and ensuring they are properly referenced, counted and loaded. It handles various types of assets, including fonts, textures, and sounds. Inspiration for asset-loading techniques can be drawn from the Unity engine, particularly the use of meta files to load assets and associated data.

### Scene Manager
The Scene Manager manages and updates scenes, their entities, and all the components. This class holds all the Scenes. Each scene contains a collection of entities, and continuously updates and removes entities during the game loop. 
The Scene Manager supports loading multiple scenes but always has at least one scene available to run. 
When requested to load a scene, the Scene Manager ensures that the scene, its entities, and components are properly loaded and initialized. 
Entities are created exclusively through the Scene Manager. 
Components are attached to entities, with the Transform component being a mandatory presence in every entity. Other components serve specific purposes and can be present or absent based on the desired entity's behaviour.
A component can only exist as part of an entity. The base component class is abstract and is designed to be derived.

### Render System
The Render System is responsible for managing the render window and all rendering-related tasks. The Renderer class holds the Renderable components and executes the rendering process. It does not handle the Initialize/Update/Destroy lifecycle methods of the components, as those responsibilities lie with the Scene Manager class. A Renderable is an abstract class representing a component with visual elements. Components such as Sprite, AnimatedSprite, and Font are implemented as Renderables.

### Collision System
The Collision System is a basic 2D collision system that utilizes the Box2D for collision detection. It manages collision checking between every entity in the game world that has Collider components. The Collision System handles the registration and removal of colliders. Currently, the system manually checks collisions between all colliders without considering the context. 

This project is still under development and may undergo further refinements as the development process progresses.