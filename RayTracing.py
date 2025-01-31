#Import pygame library and initialize
import pygame
import sys
import math
import random
import copy
from pygame.locals import QUIT

pygame.init()
pygame.mixer.init()


# Declare Surface
screenW = 400
screenH = 400
surface = pygame.display.set_mode((screenW, screenH))

resolution = 200
maxBounce = 6
raysPerPixel = 1

numRenderedFrames = 0
PI = 3.141592653589793238462643383279502884197
keyPresses = []
rayList = []
storedRayList = []
sphereList = []
triangleList = []
meshInfoList = []
previousPixelColour = [[[0, 0, 0] for y in range(resolution)] for x in range(resolution)]

# Raytracing properties
class HitInfo:
  def __init__(self, didHit, dist, hitPos, normal):
    self.didHit = didHit
    self.dist = dist
    self.hitPos = hitPos
    self.normal = normal

class Material:
  def __init__(self, colour, emitColour, emitStrength, smoothness):
    self.colour = colour
    self.emitColour = emitColour
    self.emitStrength = emitStrength
    self.smoothness = smoothness


# Vector math functions
def Magnitude(vector):
  x, y, z = vector
  dist = (x**2 + y**2 + z**2) ** 0.5
  return(dist)

def DotProduct(vector1, vector2):
  return (vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2])

def CrossProduct(vector1, vector2):
  x1, y1, z1 = vector1
  x2, y2, z2 = vector2
  cx = y1*z2 - z1*y2
  cy = z1*x2 - x1*z2
  cz = x1*y2 - y1*x2
  return([cx, cy, cz])
  
def AngleBetween(vector1, vector2):
  theta = math.acos(DotProduct(vector1, vector2) / (Magnitude(vector1) * Magnitude(vector2)))
  return theta

def AddVector(vector1, vector2):
  return [vector1[0] + vector2[0], vector1[1] + vector2[1], vector1[2] + vector2[2]]

def SubtractVector(vector1, vector2):
  return [vector1[0] - vector2[0], vector1[1] - vector2[1], vector1[2] - vector2[2]]

def MultiplyVector(vector1, vector2):
  return [vector1[0] * vector2[0], vector1[1] * vector2[1], vector1[2] * vector2[2]]

def DivideVector(vector1, vector2):
  return [vector1[0] / vector2[0], vector1[1] / vector2[1], vector1[2] / vector2[2]]

def MultiplyScalar(vector, scalar):
  return [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]

def NormalizeVector(vector):
  dX, dY, dZ = vector
  dist = Magnitude(vector)
  return [dX/dist, dY/dist, dZ/dist]

def RotateVectorX(vector, xRot):
  x, y, z = vector
  newX = x
  newY = y * math.cos(xRot) - z * math.sin(xRot)
  newZ = y * math.sin(xRot) + z * math.cos(xRot)
  return([newX, newY, newZ])
  
def RotateVectorY(vector, yRot):
  x, y, z = vector
  newX = x * math.cos(yRot) + z * math.sin(yRot)
  newY = y
  newZ = -x * math.sin(yRot) + z * math.cos(yRot)
  return([newX, newY, newZ])

def RotateVectorZ(vector, zRot):
  x, y, z = vector
  newX = x * math.cos(zRot) - y * math.sin(zRot)
  newY = x * math.sin(zRot) + y * math.cos(zRot)
  newZ = z
  return([newX, newY, newZ])

def FindRotation(rot):
  dX, dY, dZ = rot
  xRot = math.asin(dZ)
  yRot = math.atan2(dY, dX)
  
  return [xRot, yRot]

def ReflectVector(vector, normal):
  # v = vector
  # n = normal
  # k = 2.0*(v[0]*n[0] + v[1]*n[1] + v[2]*n[2])
  # v1 = [0, 0, 0]
  # v1[0] = v[0] - k*n[0]
  # v1[1] = v[1] - k*n[1]
  # v1[2] = v[2] - k*n[2]
  # return v1
  return NormalizeVector(SubtractVector(vector, MultiplyScalar(normal, 2 * DotProduct(vector, normal))))

def RandomNormalDistribution():
  theta = 2 * PI * random.uniform(0, 1)
  rho = math.sqrt(-2 * math.log(random.uniform(0, 1)))
  return rho * math.cos(theta)
  
def RandomDirection():
  x = RandomNormalDistribution()
  y = RandomNormalDistribution()
  z = RandomNormalDistribution()
  return NormalizeVector([x, y, z])

def RandomHemisphereDirection(normal):
  dir = RandomDirection()
  return MultiplyScalar(dir, math.copysign(1, DotProduct(normal, dir)))

# Raytracing functions
def SphereCollide(ray, sphereCentre, sphereRadius):
  offsetRayOrigin = SubtractVector(ray.pos, sphereCentre)
  
  a = DotProduct(ray.dir, ray.dir) # =1
  b = 2 * DotProduct(offsetRayOrigin, ray.dir)
  c = DotProduct(offsetRayOrigin, offsetRayOrigin) - sphereRadius**2
  
  discriminant = b**2 - 4*a*c
  
  hitInfo = HitInfo(False, 0, [], [])
  if discriminant >= 0:
    dist = (-b - math.sqrt(discriminant)) / (2*a)
    
    if dist >= 0:
      didHit = True
      dist = dist
      hitPos = AddVector(ray.pos, MultiplyScalar(ray.dir, dist))
      normal = NormalizeVector(SubtractVector(hitPos, sphereCentre))
      hitInfo = HitInfo(didHit, dist, hitPos, normal)

  return hitInfo

def TriangleCollide(ray, tri):
  posA = tri.posA
  posB = tri.posB
  posC = tri.posC
  edgeAB = SubtractVector(posB, posA)
  edgeAC = SubtractVector(posC, posA)
  
  normalVector = CrossProduct(edgeAB, edgeAC)
  normalVector = MultiplyScalar(normalVector, math.copysign(1, -DotProduct(ray.dir, normalVector)))

  ao = SubtractVector(ray.pos, posA)
  dao = CrossProduct(ao, ray.dir)
  
  determinant = -DotProduct(ray.dir, normalVector)
  invDet = 1 / determinant

  dist = DotProduct(ao, normalVector) * invDet

  u = DotProduct(edgeAC, dao) * invDet
  v = -DotProduct(edgeAB, dao) * invDet

  didHit = (abs(determinant) >= 0.000001 and dist >= 0 and u >= 0 and v >= 0 and (u + v) <= 1)
  
  hitPos = AddVector(ray.pos, MultiplyScalar(ray.dir, dist))
  # normal = NormalizeVector(MultiplyScalar(normalVector, math.copysign(1, determinant)))
  normal = NormalizeVector(normalVector)


  return HitInfo(didHit, dist, hitPos, normal)

 
def BoxCollide(ray, boxMin, boxMax):
  for i in range(3):
    if ray.dir[i] == 0:
      ray.dir[i] = 0.00000000001
  
  if ray.dir[0] >= 0:
    tMin = (boxMin[0] - ray.pos[0]) / ray.dir[0]
    tMax = (boxMax[0] - ray.pos[0]) / ray.dir[0]
  else:
    tMin = (boxMax[0] - ray.pos[0]) / ray.dir[0]
    tMax = (boxMin[0] - ray.pos[0]) / ray.dir[0]
  if ray.dir[1] >= 0:
    tyMin = (boxMin[1] - ray.pos[1]) / ray.dir[1]
    tyMax = (boxMax[1] - ray.pos[1]) / ray.dir[1]
  else:
    tyMin = (boxMax[1] - ray.pos[1]) / ray.dir[1]
    tyMax = (boxMin[1] - ray.pos[1]) / ray.dir[1]
  
  if (tMin > tyMax) or (tyMin > tMax):
    return False
  
  tMin = max(tMin, tyMin)
  tMax = min(tMax, tyMax)
  
  if ray.dir[2] >= 0:
    tzMin = (boxMin[2] - ray.pos[2]) / ray.dir[2]
    tzMax = (boxMax[2] - ray.pos[2]) / ray.dir[2]
  else:
    tzMin = (boxMax[2] - ray.pos[2]) / ray.dir[2]
    tzMax = (boxMin[2] - ray.pos[2]) / ray.dir[2]
  
  if (tMin > tzMax) or (tzMin > tMax):
    return False
  
  return True

def CalculateRayCollision(ray):
  closestHitInfo = HitInfo(False, 999999999999, [], [])
  for i in range(len(sphereList)):
    sphere = sphereList[i]
    objectHitInfo = SphereCollide(ray, sphere.pos, sphere.radius)
    
    if objectHitInfo.didHit and objectHitInfo.dist < closestHitInfo.dist:
      closestHitInfo = objectHitInfo
      closestHitInfo.material = sphere.material

  for i in range(len(meshInfoList)):
    meshInfo = meshInfoList[i]
    if (not BoxCollide(ray, meshInfo.boundsMin, meshInfo.boundsMax)):
      continue
    
    for j in range(meshInfo.numTriangles):
      triIndex = meshInfo.firstTriangleIndex + j
      tri = triangleList[triIndex]
      objectHitInfo = TriangleCollide(ray, tri)
      
      if objectHitInfo.didHit and objectHitInfo.dist < closestHitInfo.dist:
        closestHitInfo = objectHitInfo
        closestHitInfo.material = meshInfo.material
  
  return closestHitInfo

def GetEnvironmentLight(ray):
  return([0, 0, 0])

def CalculatePixelColour(x, y):
  global numRenderedFrames, previousPixelColour
  
  totalIncomingLight = [0, 0, 0]
  for i in range(raysPerPixel):
    totalIncomingLight = AddVector(totalIncomingLight, rayList[x][y].Trace())
  pixelColour = []
  for i in range(3):
    pixelColour.append(math.sqrt(totalIncomingLight[i] / raysPerPixel))
  
  
  weight = 1 / (numRenderedFrames + 1)
  previousPixelColour[x][y] = AddVector(MultiplyScalar(previousPixelColour[x][y], 1 - weight), MultiplyScalar(pixelColour, weight))
  
  pixelColour = previousPixelColour[x][y]
  for i in range(3):
    pixelColour[i] = min(pixelColour[i], 1)

  return(MultiplyScalar(pixelColour, 255))

class Triangle:
  def __init__(self, posA, posB, posC):
    self.posA = posA
    self.posB = posB
    self.posC = posC
    
    
class MeshInfo:
  def __init__(self, firstTriangleIndex, numTriangles, boundsMin, boundsMax, material):
    self.firstTriangleIndex = firstTriangleIndex
    self.numTriangles = numTriangles
    self.boundsMin = boundsMin
    self.boundsMax = boundsMax
    self.material = material
  
  
class Object3D:
  def __init__(self, pos, dir):
    self.pos = pos
    self.dir = dir
    # self.colour = colour
    self.xRot, self.yRot = FindRotation(NormalizeVector(dir))

  def SetPos(self, newPos):
    self.pos = newPos
  def SetDir(self, newDir):
    self.dir = newDir
    
  def Translate(self, translation):
    self.pos = AddVector(self.pos, translation)
  
class Player(Object3D):
  def __init__(self, pos, dir):
    super().__init__(pos, dir)

class Camera(Object3D):
  def __init__(self, pos, dir, focalLength):
    super().__init__(pos, dir)
    self.focalLength = focalLength

class Sphere(Object3D):
  def __init__(self, pos, radius, material):
    super().__init__(pos, [1, 0, 0])
    self.radius = radius
    self.material = material
    
class Ray(Object3D):
  def __init__(self, pos, dir):
    super().__init__(pos, dir)
    # self.hitInfo = HitInfo(False, 0, [], [])

  def Trace(self):
    incomingLight = [0, 0, 0]
    rayColour = [1, 1, 1]
    
    for i in range(maxBounce):
      self.hitInfo = CalculateRayCollision(self)
      if self.hitInfo.didHit:
        self.pos = self.hitInfo.hitPos
        material = self.hitInfo.material
        
        self.diffuseDir = NormalizeVector(AddVector(self.hitInfo.normal, RandomDirection()))
        self.specularDir = ReflectVector(self.dir, self.hitInfo.normal)
        for j in range(3):
          self.dir[j] = self.diffuseDir[j] + (self.specularDir[j] - self.diffuseDir[j]) * material.smoothness
        # self.dir = self.specularDir
        
        emittedLight = MultiplyScalar(material.emitColour, material.emitStrength)
        incomingLight = AddVector(incomingLight, MultiplyVector(emittedLight, rayColour))
        rayColour = MultiplyVector(rayColour, material.colour)
      else:
        incomingLight = AddVector(incomingLight, MultiplyVector(GetEnvironmentLight(self), rayColour))
        break
    
    #for i in range(3):
      #incomingLight[i] = math.sqrt(incomingLight[i] / raysPerPixel)
    
    return incomingLight

def GenerateRays(rayResolution):
  global rayList
  
  xDir = RotateVectorY(camera.dir, PI/2)
  yDir = CrossProduct(camera.dir, xDir)
  planePos = AddVector(camera.pos, MultiplyScalar(camera.dir, camera.focalLength))
  
  rayList = []
  for x in range(rayResolution):
    rayList.append([])
    for y in range(rayResolution):
      rayPos = []
      for i in range(3):
        rayPos.append(planePos[i] + xDir[i]*(x - rayResolution/2 + 0.5)*(screenW/rayResolution) + yDir[i]*(y - rayResolution/2 + 0.5)*(screenH/rayResolution))
      
      rayList[x].append(Ray(camera.pos, NormalizeVector(SubtractVector(rayPos, camera.pos))))
 
player = Player([0, 0, 0], [1, 0, 0])
#camera = Camera([0, 0, 0], NormalizeVector([1, 0, 1]), 800)
camera = Camera([10, -10, 10], NormalizeVector([1, 0, 1]), 200)

smoothTest = 0.99
testMaterial1 = Material([0.2, 0.2, 0.2], [1, 1, 1], 100, smoothTest)
testMaterial2 = Material([1, 0.1, 0.1], [1, 1, 1], 10, smoothTest)
testMaterial3 = Material([0.2, 1, 0.2], [0.1, 1, 0.1], 0, 0.2)
testMaterial4 = Material([0.2, 0.2, 1], [0.1, 0.1, 1], 0, 0.4)
testMaterial5 = Material([0.5, 0.52, 0.5], [1, 0, 1], 0, smoothTest)
testMaterial6 = Material([0.2, 1, 0.2], [0.1, 1, 0.1], 0, 0.6)
testMaterial7 = Material([0.2, 0.2, 1], [0.1, 0.1, 1], 0, 0.8)
#Testing
#sphereList.append(Sphere([20, 0, 9999], 8999, testMaterial1))
#sphereList.append(Sphere([20, 0, 0], 1, testMaterial2))
#sphereList.append(Sphere([20, 0, -3], 1, testMaterial3))
#sphereList.append(Sphere([20, 0, 3], 1, testMaterial4))
#sphereList.append(Sphere([20, 101, 0], 100, testMaterial5))
#meshInfoList.append(MeshInfo(0, 1, [0, 1, -99999], [99999, 1, 99999], testMaterial5))
#triangleList.append(Triangle([0, 1, 0], [99999, 1, -99999], [99999, 1, 99999]))

sphereList.append(Sphere([14, -18, 14], 4, testMaterial2))
sphereList.append(Sphere([14, -10, 26], 2, testMaterial3))
sphereList.append(Sphere([18, -10, 22], 2, testMaterial4))
sphereList.append(Sphere([22, -10, 18], 2, testMaterial6))
sphereList.append(Sphere([24, -10, 14], 2, testMaterial7))
triangleList.append(Triangle([0, 0, 0], [40, 0, 0], [40, 0, 40]))
triangleList.append(Triangle([0, 0, 0], [40, 0, 40], [0, 0, 40]))
triangleList.append(Triangle([0, 0, 0], [20, -60, 20], [40, 0, 0]))
triangleList.append(Triangle([0, 0, 0], [0, 0, 40], [20, -60, 20]))
triangleList.append(Triangle([40, 0, 40], [40, 0, 0], [20, -60, 20]))
triangleList.append(Triangle([40, 0, 40], [20, -60, 20], [0, 0, 40]))
meshInfoList.append(MeshInfo(0, len(triangleList), [0, -60, 0], [40, 0, 40], testMaterial5))


GenerateRays(resolution)
storedRayList = copy.deepcopy(rayList)

while True:
  for event in pygame.event.get():
    if event.type == pygame.KEYDOWN:
      keyPresses.append(event.key)
    elif event.type == pygame.KEYUP:
      keyPresses.remove(event.key)
    
    if event.type == QUIT:
      pygame.quit()
      sys.exit()

  # Align Camera with Player
  #player.SetDir(AddVector(player.dir, [0, -0.1, 0]))
  camera.SetPos(player.pos)
  camera.SetDir(player.dir)

  

  # surface.fill((0, 0, 0))

  #GenerateRays(resolution)
  rayList = copy.deepcopy(storedRayList)
  
  for x in range(len(rayList)):
    for y in range(len(rayList[x])):
      colour = CalculatePixelColour(x, y)
      
      pygame.draw.rect(surface, tuple(colour), (x * screenW/resolution, y * screenH/resolution, screenW/resolution, screenH/resolution))
  
  numRenderedFrames += 1
  pygame.display.flip()
